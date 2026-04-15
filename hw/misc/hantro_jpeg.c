/*
 * Hantro JPEG Decode Accelerator (emulated)
 *
 * Emulates a simple MMIO JPEG decode engine for the QEMU virt machine.
 * The guest writes JPEG source and ARGB destination DMA addresses plus
 * the input length, then writes 1 to the CONTROL register.  The device
 * performs the decode synchronously on the host and sets STATUS to DONE.
 *
 * Register map (all 32-bit, little-endian):
 *
 *   0x00  MAGIC_ID      R     0x4A504547 ("JPEG")
 *   0x04  STATUS        R     bit0=ready  bit2=done  bit3=fault
 *   0x08  CONTROL       W     write 1 to start decode
 *   0x0C  IRQ_STATUS    R/W   write 1 to ack (unused for now)
 *   0x10  INPUT_ADDR_LO W     JPEG source phys addr low 32
 *   0x14  INPUT_ADDR_HI W     JPEG source phys addr high 32
 *   0x18  INPUT_LEN     W     JPEG byte count
 *   0x1C  OUTPUT_ADDR_LO W    ARGB dest phys addr low 32
 *   0x20  OUTPUT_ADDR_HI W    ARGB dest phys addr high 32
 *   0x24  OUTPUT_WIDTH  R     decoded width  (valid after DONE)
 *   0x28  OUTPUT_HEIGHT R     decoded height (valid after DONE)
 *   0x2C  OUTPUT_STRIDE W     output row stride in pixels (0 = width)
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "hw/sysbus.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "system/address-spaces.h"
#include "system/dma.h"
#include "migration/vmstate.h"

#include <jpeglib.h>
#include <setjmp.h>

/* ── Register offsets ──────────────────────────────────────────────────── */
#define REG_MAGIC_ID       0x00
#define REG_STATUS         0x04
#define REG_CONTROL        0x08
#define REG_IRQ_STATUS     0x0C
#define REG_INPUT_ADDR_LO  0x10
#define REG_INPUT_ADDR_HI  0x14
#define REG_INPUT_LEN      0x18
#define REG_OUTPUT_ADDR_LO 0x1C
#define REG_OUTPUT_ADDR_HI 0x20
#define REG_OUTPUT_WIDTH   0x24
#define REG_OUTPUT_HEIGHT  0x28
#define REG_OUTPUT_STRIDE  0x2C

#define HANTRO_MAGIC       0x4A504547  /* "JPEG" */

/* Status bits */
#define STATUS_READY       (1 << 0)
#define STATUS_BUSY        (1 << 1)
#define STATUS_DONE        (1 << 2)
#define STATUS_FAULT       (1 << 3)

/* Max JPEG input we'll DMA-read from guest (8 MiB) */
#define MAX_JPEG_INPUT     (8 * 1024 * 1024)
/* Max decoded output (3840×2160×4 = 33 MiB) */
#define MAX_OUTPUT_PIXELS  (3840 * 2160)

#define TYPE_HANTRO_JPEG   "hantro-jpeg-dec"

typedef struct HantroJpegState {
    SysBusDevice parent_obj;

    MemoryRegion mmio;

    /* Registers */
    uint32_t status;
    uint32_t irq_status;
    uint32_t input_addr_lo;
    uint32_t input_addr_hi;
    uint32_t input_len;
    uint32_t output_addr_lo;
    uint32_t output_addr_hi;
    uint32_t output_width;
    uint32_t output_height;
    uint32_t output_stride;

    /* Host-side buffers (allocated once, reused) */
    uint8_t *jpeg_buf;
    uint32_t *argb_buf;
    size_t jpeg_buf_size;
    size_t argb_buf_size;  /* in pixels */
} HantroJpegState;

#define HANTRO_JPEG(obj) \
    OBJECT_CHECK(HantroJpegState, (obj), TYPE_HANTRO_JPEG)

/* ── libjpeg error handler that doesn't abort() ─────────────────────── */
struct hantro_jpeg_error {
    struct jpeg_error_mgr pub;
    jmp_buf jmpbuf;
};

static void hantro_jpeg_error_exit(j_common_ptr cinfo)
{
    struct hantro_jpeg_error *err = (struct hantro_jpeg_error *)cinfo->err;
    longjmp(err->jmpbuf, 1);
}

/* ── Host-side JPEG decode: JPEG → ARGB8888 ────────────────────────── */
static int hantro_do_decode(HantroJpegState *s)
{
    struct jpeg_decompress_struct cinfo;
    struct hantro_jpeg_error jerr;
    uint64_t input_addr, output_addr;
    uint32_t w, h, stride;
    JSAMPROW row_ptr;
    uint8_t *rgb_row = NULL;

    input_addr = ((uint64_t)s->input_addr_hi << 32) | s->input_addr_lo;
    output_addr = ((uint64_t)s->output_addr_hi << 32) | s->output_addr_lo;

    if (s->input_len == 0 || s->input_len > MAX_JPEG_INPUT) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "hantro-jpeg: invalid input_len %u\n", s->input_len);
        return -1;
    }

    /* Ensure host buffers are large enough */
    if (s->jpeg_buf_size < s->input_len) {
        s->jpeg_buf = g_realloc(s->jpeg_buf, s->input_len);
        s->jpeg_buf_size = s->input_len;
    }

    /* DMA-read JPEG data from guest physical memory */
    if (dma_memory_read(&address_space_memory, input_addr,
                        s->jpeg_buf, s->input_len, MEMTXATTRS_UNSPECIFIED)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "hantro-jpeg: DMA read failed at 0x%" PRIx64 "\n",
                      input_addr);
        return -1;
    }

    /* Set up libjpeg with our custom error handler */
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = hantro_jpeg_error_exit;

    if (setjmp(jerr.jmpbuf)) {
        jpeg_destroy_decompress(&cinfo);
        g_free(rgb_row);
        return -1;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, s->jpeg_buf, s->input_len);
    jpeg_read_header(&cinfo, TRUE);

    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);

    w = cinfo.output_width;
    h = cinfo.output_height;
    stride = s->output_stride ? s->output_stride : w;

    if ((size_t)w * h > MAX_OUTPUT_PIXELS) {
        jpeg_destroy_decompress(&cinfo);
        qemu_log_mask(LOG_GUEST_ERROR,
                      "hantro-jpeg: image too large %ux%u\n", w, h);
        return -1;
    }

    /* Ensure ARGB buffer can hold the full image at stride */
    size_t total_pixels = (size_t)stride * h;
    if (s->argb_buf_size < total_pixels) {
        s->argb_buf = g_realloc(s->argb_buf, total_pixels * 4);
        s->argb_buf_size = total_pixels;
    }

    /* Allocate one row of RGB888 for scanline conversion */
    rgb_row = g_malloc(w * 3);

    /* Decompress row by row: RGB888 → ARGB8888 */
    for (uint32_t y = 0; y < h; y++) {
        row_ptr = rgb_row;
        jpeg_read_scanlines(&cinfo, &row_ptr, 1);

        uint32_t *dst = s->argb_buf + (size_t)y * stride;
        for (uint32_t x = 0; x < w; x++) {
            uint8_t r = rgb_row[x * 3 + 0];
            uint8_t g = rgb_row[x * 3 + 1];
            uint8_t b = rgb_row[x * 3 + 2];
            dst[x] = 0xFF000000u | ((uint32_t)r << 16) |
                     ((uint32_t)g << 8) | (uint32_t)b;
        }
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    g_free(rgb_row);

    /* DMA-write ARGB pixels back to guest physical memory */
    if (dma_memory_write(&address_space_memory, output_addr,
                         s->argb_buf, total_pixels * 4,
                         MEMTXATTRS_UNSPECIFIED)) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "hantro-jpeg: DMA write failed at 0x%" PRIx64 "\n",
                      output_addr);
        return -1;
    }

    s->output_width = w;
    s->output_height = h;
    return 0;
}

/* ── MMIO read ─────────────────────────────────────────────────────── */
static uint64_t hantro_jpeg_read(void *opaque, hwaddr offset, unsigned size)
{
    HantroJpegState *s = HANTRO_JPEG(opaque);

    switch (offset) {
    case REG_MAGIC_ID:
        return HANTRO_MAGIC;
    case REG_STATUS:
        return s->status;
    case REG_IRQ_STATUS:
        return s->irq_status;
    case REG_OUTPUT_WIDTH:
        return s->output_width;
    case REG_OUTPUT_HEIGHT:
        return s->output_height;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "hantro-jpeg: bad read offset 0x%" HWADDR_PRIx "\n",
                      offset);
        return 0;
    }
}

/* ── MMIO write ────────────────────────────────────────────────────── */
static void hantro_jpeg_write(void *opaque, hwaddr offset,
                              uint64_t value, unsigned size)
{
    HantroJpegState *s = HANTRO_JPEG(opaque);

    switch (offset) {
    case REG_CONTROL:
        if (value & 1) {
            /* Start decode */
            s->status = STATUS_BUSY;
            if (hantro_do_decode(s) == 0) {
                s->status = STATUS_READY | STATUS_DONE;
            } else {
                s->status = STATUS_READY | STATUS_FAULT;
            }
        }
        break;
    case REG_IRQ_STATUS:
        s->irq_status &= ~(uint32_t)value;
        break;
    case REG_INPUT_ADDR_LO:
        s->input_addr_lo = (uint32_t)value;
        break;
    case REG_INPUT_ADDR_HI:
        s->input_addr_hi = (uint32_t)value;
        break;
    case REG_INPUT_LEN:
        s->input_len = (uint32_t)value;
        break;
    case REG_OUTPUT_ADDR_LO:
        s->output_addr_lo = (uint32_t)value;
        break;
    case REG_OUTPUT_ADDR_HI:
        s->output_addr_hi = (uint32_t)value;
        break;
    case REG_OUTPUT_STRIDE:
        s->output_stride = (uint32_t)value;
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "hantro-jpeg: bad write offset 0x%" HWADDR_PRIx "\n",
                      offset);
        break;
    }
}

static const MemoryRegionOps hantro_jpeg_ops = {
    .read = hantro_jpeg_read,
    .write = hantro_jpeg_write,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* ── Device init / reset ───────────────────────────────────────────── */
static void hantro_jpeg_reset(DeviceState *dev)
{
    HantroJpegState *s = HANTRO_JPEG(dev);

    s->status = STATUS_READY;
    s->irq_status = 0;
    s->input_addr_lo = 0;
    s->input_addr_hi = 0;
    s->input_len = 0;
    s->output_addr_lo = 0;
    s->output_addr_hi = 0;
    s->output_width = 0;
    s->output_height = 0;
    s->output_stride = 0;
}

static void hantro_jpeg_init(Object *obj)
{
    HantroJpegState *s = HANTRO_JPEG(obj);
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);

    memory_region_init_io(&s->mmio, obj, &hantro_jpeg_ops, s,
                          TYPE_HANTRO_JPEG, 0x1000);
    sysbus_init_mmio(sbd, &s->mmio);

    /* Pre-allocate host decode buffers */
    s->jpeg_buf_size = 512 * 1024;  /* 512 KiB initial */
    s->jpeg_buf = g_malloc(s->jpeg_buf_size);
    s->argb_buf_size = 1920 * 1080;
    s->argb_buf = g_malloc(s->argb_buf_size * 4);
}

static void hantro_jpeg_finalize(Object *obj)
{
    HantroJpegState *s = HANTRO_JPEG(obj);

    g_free(s->jpeg_buf);
    g_free(s->argb_buf);
    s->jpeg_buf = NULL;
    s->argb_buf = NULL;
}

static const VMStateDescription hantro_jpeg_vmstate = {
    .name = TYPE_HANTRO_JPEG,
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_UINT32(status, HantroJpegState),
        VMSTATE_UINT32(input_addr_lo, HantroJpegState),
        VMSTATE_UINT32(input_addr_hi, HantroJpegState),
        VMSTATE_UINT32(input_len, HantroJpegState),
        VMSTATE_UINT32(output_addr_lo, HantroJpegState),
        VMSTATE_UINT32(output_addr_hi, HantroJpegState),
        VMSTATE_UINT32(output_width, HantroJpegState),
        VMSTATE_UINT32(output_height, HantroJpegState),
        VMSTATE_UINT32(output_stride, HantroJpegState),
        VMSTATE_END_OF_LIST()
    },
};

static void hantro_jpeg_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);

    device_class_set_legacy_reset(dc, hantro_jpeg_reset);
    dc->vmsd = &hantro_jpeg_vmstate;
    dc->desc = "Hantro JPEG Decode Accelerator (emulated)";
}

static const TypeInfo hantro_jpeg_types[] = {
    {
        .name          = TYPE_HANTRO_JPEG,
        .parent        = TYPE_SYS_BUS_DEVICE,
        .instance_size = sizeof(HantroJpegState),
        .instance_init = hantro_jpeg_init,
        .instance_finalize = hantro_jpeg_finalize,
        .class_init    = hantro_jpeg_class_init,
    }
};

DEFINE_TYPES(hantro_jpeg_types);
