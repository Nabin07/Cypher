/*
 * CYPHER native — RK3568 EVB
 * Framebuffer UI + ALSA audio + Touch input
 * gcc -O2 -o cypher main.c sub808.c kick.c synth.c reverb.c -lasound -lm -lpthread
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/fb.h>
#include <linux/input.h>
#include <alsa/asoundlib.h>

#include "types.h"
#include "sub808.h"
#include "kick.h"
#include "synth.h"
#include "reverb.h"
#include "chords.h"
#include "sampler.h"

/* ── Display ── */
/* FB is 1080x1920 portrait in memory. We do manual 270-deg rotation:
   landscape (1920x1080) -> portrait FB. */
#define FB_W 1080
#define FB_H 1920
#define FB_STRIDE 4352
#define SCREEN_W 1920
#define SCREEN_H 1080

/* ── Colors (XRGB8888: B,G,R,X in memory) ── */
#define COL_BG       0xFF121224
#define COL_PANEL    0xFF1C1C32
#define COL_TEXT     0xFFC8B4B4
#define COL_DIM      0xFF645050
#define COL_CYAN     0xFFFFD200
#define COL_GREEN    0xFF82FF00
#define COL_RED      0xFF4444FF
#define COL_YELLOW   0xFF00D7FF
#define COL_MAGENTA  0xFFFF64FF
#define COL_SLIDER   0xFF413028

/* ── Engine state ── */
enum { ENG_808, ENG_KICK, ENG_SYNTH, ENG_SAMPLER, ENG_FX, ENG_CHORD, ENG_COUNT };
static const char *ENG_NAMES[] = {"808", "KICK", "SYNTH", "SMPLR", "FX", "CHORD"};

typedef struct {
    Sub808 sub808;
    Kick kick;
    PolySynth synth;
    SamplerEngine sampler;
    Reverb reverb;

    int engine_idx;
    int octave;
    int selected_param;
    int current_page;
    int reverb_on;
    float fx_send[3]; /* send amount per engine */
    int fx_on[3];     /* send on/off per engine */
    float peak;

    /* Chord engine state */
    int chord_prog_idx;
    int chord_step;
    int chord_mode; /* 0=chord, 1=strum dn, 2=strum up, 3=arp */

    /* Sampler edit state */
    int selected_slice;

    pthread_mutex_t lock;
    volatile int running;

    /* Framebuffer */
    uint32_t *fb;
    int fb_fd;

    /* Touch */
    int touch_fd;
    int touch_x, touch_y, touch_down;
    struct timespec last_tap;

    /* MIDI */
    int midi_fd;
    uint8_t active_notes[128]; /* 1 if note is on */
} App;

/* ── Framebuffer helpers ── */
static inline void fb_pixel(App *app, int sx, int sy, uint32_t col) {
    /* Landscape (sx,sy) where sx=0..1919, sy=0..1079
       270-deg rotation to portrait FB: fb_x = sy, fb_y = FB_H-1-sx */
    int fx = sy;
    int fy = FB_H - 1 - sx;
    if (fx < 0 || fx >= FB_W || fy < 0 || fy >= FB_H) return;
    uint32_t *row = (uint32_t *)((uint8_t *)app->fb + fy * FB_STRIDE);
    row[fx] = col;
}

static void fb_rect(App *app, int x, int y, int w, int h, uint32_t col) {
    for (int dy = 0; dy < h; dy++)
        for (int dx = 0; dx < w; dx++)
            fb_pixel(app, x + dx, y + dy, col);
}

static void fb_char(App *app, int x, int y, char c, uint32_t col);
static void fb_text(App *app, int x, int y, const char *s, uint32_t col) {
    while (*s) { fb_char(app, x, y, *s, col); x += 16; s++; }
}

/* Minimal 8x8 font - just uppercase + digits + symbols */
static const uint8_t FONT[][8] = {
    /* space */ {0,0,0,0,0,0,0,0},
    /* A */ {0x3C,0x66,0x66,0x7E,0x66,0x66,0x66,0x00},
    /* B */ {0x7C,0x66,0x66,0x7C,0x66,0x66,0x7C,0x00},
    /* C */ {0x3C,0x66,0x60,0x60,0x60,0x66,0x3C,0x00},
    /* D */ {0x78,0x6C,0x66,0x66,0x66,0x6C,0x78,0x00},
    /* E */ {0x7E,0x60,0x60,0x7C,0x60,0x60,0x7E,0x00},
    /* F */ {0x7E,0x60,0x60,0x7C,0x60,0x60,0x60,0x00},
    /* G */ {0x3C,0x66,0x60,0x6E,0x66,0x66,0x3E,0x00},
    /* H */ {0x66,0x66,0x66,0x7E,0x66,0x66,0x66,0x00},
    /* I */ {0x3C,0x18,0x18,0x18,0x18,0x18,0x3C,0x00},
    /* J */ {0x06,0x06,0x06,0x06,0x66,0x66,0x3C,0x00},
    /* K */ {0x66,0x6C,0x78,0x70,0x78,0x6C,0x66,0x00},
    /* L */ {0x60,0x60,0x60,0x60,0x60,0x60,0x7E,0x00},
    /* M */ {0xC6,0xEE,0xFE,0xD6,0xC6,0xC6,0xC6,0x00},
    /* N */ {0x66,0x76,0x7E,0x7E,0x6E,0x66,0x66,0x00},
    /* O */ {0x3C,0x66,0x66,0x66,0x66,0x66,0x3C,0x00},
    /* P */ {0x7C,0x66,0x66,0x7C,0x60,0x60,0x60,0x00},
    /* Q */ {0x3C,0x66,0x66,0x66,0x6A,0x6C,0x36,0x00},
    /* R */ {0x7C,0x66,0x66,0x7C,0x6C,0x66,0x66,0x00},
    /* S */ {0x3C,0x66,0x60,0x3C,0x06,0x66,0x3C,0x00},
    /* T */ {0x7E,0x18,0x18,0x18,0x18,0x18,0x18,0x00},
    /* U */ {0x66,0x66,0x66,0x66,0x66,0x66,0x3C,0x00},
    /* V */ {0x66,0x66,0x66,0x66,0x66,0x3C,0x18,0x00},
    /* W */ {0xC6,0xC6,0xC6,0xD6,0xFE,0xEE,0xC6,0x00},
    /* X */ {0x66,0x66,0x3C,0x18,0x3C,0x66,0x66,0x00},
    /* Y */ {0x66,0x66,0x66,0x3C,0x18,0x18,0x18,0x00},
    /* Z */ {0x7E,0x06,0x0C,0x18,0x30,0x60,0x7E,0x00},
    /* 0 */ {0x3C,0x66,0x6E,0x76,0x66,0x66,0x3C,0x00},
    /* 1 */ {0x18,0x38,0x18,0x18,0x18,0x18,0x7E,0x00},
    /* 2 */ {0x3C,0x66,0x06,0x0C,0x18,0x30,0x7E,0x00},
    /* 3 */ {0x3C,0x66,0x06,0x1C,0x06,0x66,0x3C,0x00},
    /* 4 */ {0x0C,0x1C,0x3C,0x6C,0x7E,0x0C,0x0C,0x00},
    /* 5 */ {0x7E,0x60,0x7C,0x06,0x06,0x66,0x3C,0x00},
    /* 6 */ {0x3C,0x66,0x60,0x7C,0x66,0x66,0x3C,0x00},
    /* 7 */ {0x7E,0x06,0x0C,0x18,0x30,0x30,0x30,0x00},
    /* 8 */ {0x3C,0x66,0x66,0x3C,0x66,0x66,0x3C,0x00},
    /* 9 */ {0x3C,0x66,0x66,0x3E,0x06,0x66,0x3C,0x00},
    /* : */ {0x00,0x18,0x18,0x00,0x18,0x18,0x00,0x00},
    /* . */ {0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00},
    /* - */ {0x00,0x00,0x00,0x7E,0x00,0x00,0x00,0x00},
    /* / */ {0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x00},
    /* % */ {0x62,0x64,0x08,0x10,0x26,0x46,0x06,0x00},
    /* [ */ {0x3C,0x30,0x30,0x30,0x30,0x30,0x3C,0x00},
    /* ] */ {0x3C,0x0C,0x0C,0x0C,0x0C,0x0C,0x3C,0x00},
};

static int char_index(char c) {
    if (c == ' ') return 0;
    if (c >= 'A' && c <= 'Z') return 1 + (c - 'A');
    if (c >= 'a' && c <= 'z') return 1 + (c - 'a');
    if (c >= '0' && c <= '9') return 27 + (c - '0');
    if (c == ':') return 37; if (c == '.') return 38;
    if (c == '-') return 39; if (c == '/') return 40;
    if (c == '%') return 41; if (c == '[') return 42; if (c == ']') return 43;
    return 0;
}

static void fb_char(App *app, int x, int y, char c, uint32_t col) {
    int idx = char_index(c);
    const uint8_t *glyph = FONT[idx];
    /* 2x scale */
    for (int row = 0; row < 8; row++)
        for (int bit = 0; bit < 8; bit++)
            if (glyph[row] & (0x80 >> bit)) {
                fb_pixel(app, x + bit*2,   y + row*2,   col);
                fb_pixel(app, x + bit*2+1, y + row*2,   col);
                fb_pixel(app, x + bit*2,   y + row*2+1, col);
                fb_pixel(app, x + bit*2+1, y + row*2+1, col);
            }
}

/* ── MIDI keyboard drawing (1 octave, C-B) ── */
static const char *KEY_NAMES[] = {"C","D","E","F","G","A","B"};
static const int WHITE_SEMI[] = {0,2,4,5,7,9,11};
static const int BLACK_SEMI[] = {1,3,6,8,10};
static const int BLACK_POS[] = {0,1,3,4,5}; /* which white key each black is after */

static void draw_keyboard(App *app, int y) {
    int key_w = 120, key_h = 180;
    int start_note = app->octave * 12 + 24;
    int x = 200;

    /* White keys */
    for (int i = 0; i < 7; i++) {
        int note = start_note + WHITE_SEMI[i];
        int active = (note < 128) ? app->active_notes[note] : 0;
        uint32_t col = active ? COL_GREEN : 0xFFE0E0E0;
        fb_rect(app, x + i * key_w, y, key_w - 4, key_h, col);
        /* Note name */
        fb_text(app, x + i * key_w + 40, y + key_h - 30, KEY_NAMES[i], 0xFF404040);
    }

    /* Black keys */
    for (int i = 0; i < 5; i++) {
        int note = start_note + BLACK_SEMI[i];
        int active = (note < 128) ? app->active_notes[note] : 0;
        uint32_t col = active ? COL_GREEN : 0xFF201818;
        int bx = x + BLACK_POS[i] * key_w + key_w - 32;
        fb_rect(app, bx, y, 64, 110, col);
    }
}

/* ── UI draw ── */
static void draw_ui(App *app) {
    /* Clear */
    fb_rect(app, 0, 0, SCREEN_W, SCREEN_H, COL_BG);

    /* Title */
    char title[64];
    snprintf(title, sizeof(title), "CYPHER %s", ENG_NAMES[app->engine_idx]);
    fb_text(app, SCREEN_W / 2 - 120, 20, title, COL_CYAN);

    /* Engine tabs */
    int tx = 80;
    for (int i = 0; i < ENG_COUNT; i++) {
        uint32_t col = (i == app->engine_idx) ? COL_CYAN : COL_DIM;
        char tab[16];
        snprintf(tab, sizeof(tab), "[%s]", ENG_NAMES[i]);
        fb_text(app, tx, 70, tab, col);
        tx += 200;
    }

    /* Level meter */
    fb_text(app, 80, 120, "LEVEL", COL_TEXT);
    fb_rect(app, 220, 124, 1200, 20, COL_SLIDER);
    int lw = (int)(app->peak * 1200);
    if (lw > 0) fb_rect(app, 220, 124, lw < 1200 ? lw : 1200, 20, lw < 960 ? COL_GREEN : COL_RED);

    /* ── SAMPLER tab ── */
    if (app->engine_idx == ENG_SAMPLER) {
        SampleSlot *fsl = &app->sampler.slots[app->sampler.focused_slot];
        sampler_ensure_slices(&app->sampler, app->sampler.focused_slot);
        if (app->selected_slice < 0 ||
            (fsl->slice_count > 0 && app->selected_slice >= fsl->slice_count))
            app->selected_slice = 0;

        /* 4x4 pad grid */
        int pad_w = 100, pad_h = 80, pad_x0 = 80, pad_y0 = 170;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                int idx = r * 4 + c;
                int px = pad_x0 + c * (pad_w + 10);
                int py = pad_y0 + r * (pad_h + 10);
                SampleSlot *sl = &app->sampler.slots[idx];
                int focused = idx == app->sampler.focused_slot;
                uint32_t bg = sl->loaded ? (focused ? 0xFF60B880 : 0xFF408260) : (focused ? 0xFF505070 : COL_PANEL);
                fb_rect(app, px, py, pad_w, pad_h, bg);
                char num[4]; snprintf(num, sizeof(num), "%d", idx + 1);
                fb_text(app, px + 8, py + 8, num, COL_TEXT);
                if (sl->loaded) {
                    char nm[12];
                    strncpy(nm, sl->name, 8); nm[8] = 0;
                    fb_text(app, px + 8, py + 40, nm, 0xFFE0E0F0);
                }
            }
        }

        /* Focused slot params (right side) */
        int num_pages = (SAMPLER_SLOT_PARAMS + 3) / 4;
        if (app->current_page >= num_pages) app->current_page = 0;
        int ps = app->current_page * 4;
        int pe = ps + 4; if (pe > SAMPLER_SLOT_PARAMS) pe = SAMPLER_SLOT_PARAMS;

        char pg[32]; snprintf(pg, sizeof(pg), "PAD %d  PAGE %d/%d", app->sampler.focused_slot + 1, app->current_page + 1, num_pages);
        fb_text(app, 560, 170, pg, COL_CYAN);

        fb_rect(app, 540, 200, 900, 360, COL_PANEL);
        for (int i = 0; i < pe - ps; i++) {
            int py = 220 + i * 80;
            Param *p = &fsl->params[ps + i];
            uint32_t col = (i == app->selected_param) ? COL_YELLOW : COL_TEXT;
            fb_text(app, 560, py, p->label, col);
            char val[32]; float m = param_mapped(p);
            if (strcmp(p->unit, "Hz") == 0) snprintf(val, sizeof(val), "%.0fHZ", m);
            else if (strcmp(p->unit, "ms") == 0) snprintf(val, sizeof(val), "%.0fMS", m);
            else if (strcmp(p->unit, "st") == 0) snprintf(val, sizeof(val), "%.1fST", m);
            else snprintf(val, sizeof(val), "%.2f", p->value);
            fb_text(app, 800, py, val, col);
            fb_rect(app, 1000, py + 4, 400, 16, COL_SLIDER);
            int sw = (int)(p->value * 400);
            if (sw > 0) fb_rect(app, 1000, py + 4, sw, 16, i == app->selected_param ? COL_GREEN : COL_DIM);
        }

        /* ── Slice editor ── */
        int strip_x = 80, strip_y = 605, strip_w = 1360, strip_h = 70;
        char shdr[64];
        snprintf(shdr, sizeof(shdr), "SLICES %d/%d%s",
                 fsl->slice_count > 0 ? app->selected_slice + 1 : 0,
                 fsl->slice_count,
                 fsl->slice_manual ? "  EDITED" : "");
        fb_text(app, strip_x, 575, shdr, COL_CYAN);
        fb_text(app, 700, 575, "[PREV]", COL_DIM);
        fb_text(app, 830, 575, "[NEXT]", COL_DIM);
        fb_text(app, 960, 575, "[RESET]", COL_DIM);

        fb_rect(app, strip_x, strip_y, strip_w, strip_h, 0xFF101018);
        if (fsl->loaded && fsl->length > 0 && fsl->slice_count > 0) {
            for (int i = 0; i < fsl->slice_count; i++) {
                int sx0 = strip_x + (int)((float)fsl->slice_starts[i] / fsl->length * strip_w);
                int sx1 = strip_x + (int)((float)fsl->slice_ends[i]   / fsl->length * strip_w);
                int sw = sx1 - sx0; if (sw < 1) sw = 1;
                int sel = i == app->selected_slice;
                uint32_t fill = sel ? 0xFF285028 : ((i & 1) ? 0xFF1A1A28 : 0xFF22222F);
                fb_rect(app, sx0, strip_y, sw, strip_h, fill);
                fb_rect(app, sx0, strip_y, 2, strip_h, sel ? 0xFF80FF80 : 0xFF505070);
                /* Slice number */
                if (sw > 24) {
                    char num[4]; snprintf(num, sizeof(num), "%d", i + 1);
                    fb_text(app, sx0 + 6, strip_y + 6, num, sel ? COL_GREEN : COL_DIM);
                }
            }
            int lastEnd = fsl->slice_ends[fsl->slice_count - 1];
            int sx_end = strip_x + (int)((float)lastEnd / fsl->length * strip_w);
            if (sx_end > strip_x + strip_w - 2) sx_end = strip_x + strip_w - 2;
            fb_rect(app, sx_end, strip_y, 2, strip_h, 0xFF505070);
        } else {
            fb_text(app, strip_x + 12, strip_y + 28, "LOAD SAMPLE TO EDIT SLICES", COL_DIM);
        }

        /* Start / End sliders for selected slice */
        if (fsl->loaded && fsl->slice_count > 0) {
            float sf = (float)fsl->slice_starts[app->selected_slice] / fsl->length;
            float ef = (float)fsl->slice_ends[app->selected_slice]   / fsl->length;
            char sl[24]; snprintf(sl, sizeof(sl), "START %4.1f%%", sf * 100);
            char el[24]; snprintf(el, sizeof(el), "END   %4.1f%%", ef * 100);
            fb_text(app, 80, 700, sl, COL_TEXT);
            fb_rect(app, 320, 705, 380, 18, COL_SLIDER);
            fb_rect(app, 320, 705, (int)(sf * 380), 18, COL_GREEN);
            fb_text(app, 760, 700, el, COL_TEXT);
            fb_rect(app, 1000, 705, 380, 18, COL_SLIDER);
            fb_rect(app, 1000, 705, (int)(ef * 380), 18, COL_GREEN);
        }

        /* Info */
        int loaded_count = 0, active_count = 0;
        for (int i = 0; i < 16; i++) if (app->sampler.slots[i].loaded) loaded_count++;
        for (int i = 0; i < 8; i++) if (app->sampler.voices[i].active) active_count++;
        char info[64];
        snprintf(info, sizeof(info), "SLOTS: %d/16  VOICES: %d/8", loaded_count, active_count);
        fb_text(app, 80, SCREEN_H - 80, info, COL_DIM);
    }

    /* ── CHORD tab ── */
    else if (app->engine_idx == ENG_CHORD) {
        fb_text(app, 80, 170, "PROGRESSION", COL_TEXT);
        /* Progression list */
        fb_rect(app, 60, 200, 900, 300, COL_PANEL);
        for (int i = 0; i < NUM_PROGRESSIONS; i++) {
            int py = 210 + i * 48;
            int sel = i == app->chord_prog_idx;
            uint32_t col = sel ? COL_CYAN : COL_DIM;
            if (sel) fb_rect(app, 70, py, 880, 40, 0xFF2D2D50);
            char row[128];
            /* Build chord labels */
            int root = app->octave * 12 + 24;
            int notes[5]; char lbl[16];
            char labels[64] = "";
            for (int s = 0; s < prog_length(i); s++) {
                build_prog_chord(root, i, s, notes, lbl, sizeof(lbl));
                if (s > 0) strcat(labels, " - ");
                strcat(labels, lbl);
            }
            snprintf(row, sizeof(row), "%s%s  %s", sel ? "> " : "  ", PROGRESSIONS[i].name, labels);
            fb_text(app, 80, py + 8, row, col);
        }

        /* Steps */
        fb_text(app, 1000, 200, "STEPS", COL_TEXT);
        int root = app->octave * 12 + 24;
        int plen = prog_length(app->chord_prog_idx);
        for (int si = 0; si < plen; si++) {
            int notes[5]; char lbl[16];
            build_prog_chord(root, app->chord_prog_idx, si, notes, lbl, sizeof(lbl));
            uint32_t col = (si == app->chord_step) ? COL_CYAN : COL_DIM;
            char step[24]; snprintf(step, sizeof(step), "[%s]", lbl);
            fb_text(app, 1000, 240 + si * 50, step, col);
        }

        /* Mode */
        const char *modes[] = {"CHORD", "STRUM DN", "STRUM UP", "ARP"};
        fb_text(app, 1000, 460, "MODE", COL_TEXT);
        for (int i = 0; i < 4; i++) {
            uint32_t col = (i == app->chord_mode) ? COL_GREEN : COL_DIM;
            fb_text(app, 1000, 500 + i * 40, modes[i], col);
        }
    }

    /* ── FX tab ── */
    else if (app->engine_idx == ENG_FX) {
        fb_text(app, 80, 170, "REVERB", COL_TEXT);
        fb_text(app, 300, 170, app->reverb_on ? "[ON]" : "[OFF]", app->reverb_on ? COL_GREEN : COL_RED);

        /* FX send per engine */
        fb_text(app, 80, 220, "SENDS:", COL_TEXT);
        const char *eng_names[] = {"808", "KICK", "SYNTH"};
        for (int i = 0; i < 3; i++) {
            int sy = 260 + i * 50;
            fb_text(app, 80, sy, eng_names[i], COL_TEXT);
            fb_text(app, 250, sy, app->fx_on[i] ? "ON" : "OFF", app->fx_on[i] ? COL_GREEN : COL_DIM);
            fb_rect(app, 400, sy + 4, 400, 16, COL_SLIDER);
            int sw = (int)(app->fx_send[i] * 400);
            if (sw > 0) fb_rect(app, 400, sy + 4, sw, 16, COL_GREEN);
            char pct[16]; snprintf(pct, sizeof(pct), "%d%%", (int)(app->fx_send[i] * 100));
            fb_text(app, 820, sy, pct, COL_TEXT);
        }

        fb_text(app, 80, 440, "REVERB MODE:", COL_TEXT);
        const char *rev_modes[] = {"ROOM", "CHAMBER", "HALL", "PLATE"};
        for (int i = 0; i < 4; i++) {
            uint32_t col = (i == app->reverb.mode) ? COL_GREEN : COL_DIM;
            fb_text(app, 80 + i * 200, 480, rev_modes[i], col);
        }
    }

    /* ── Normal engine UI (808/KICK/SYNTH) ── */
    else {
        Param *all_params = NULL; int total_params = 0;
        switch (app->engine_idx) {
            case ENG_808:  all_params = app->sub808.params; total_params = SUB808_PARAMS; break;
            case ENG_KICK: all_params = app->kick.params; total_params = KICK_PARAMS; break;
            case ENG_SYNTH:all_params = app->synth.params; total_params = SYNTH_PARAMS; break;
        }
        int num_pages = total_params > 0 ? (total_params + 3) / 4 : 0;
        if (app->current_page >= num_pages) app->current_page = 0;
        int ps = app->current_page * 4;
        int pe = ps + 4; if (pe > total_params) pe = total_params;

        if (num_pages > 1) {
            char pg[32]; snprintf(pg, sizeof(pg), "PAGE %d/%d", app->current_page + 1, num_pages);
            fb_text(app, 1500, 175, pg, COL_TEXT);
        }

        fb_rect(app, 60, 170, 1400, 400, COL_PANEL);
        for (int i = 0; i < pe - ps; i++) {
            int py = 190 + i * 90;
            Param *p = &all_params[ps + i];
            uint32_t col = (i == app->selected_param) ? COL_YELLOW : COL_TEXT;
            char label[32], val[32];
            snprintf(label, sizeof(label), "%s%s", i == app->selected_param ? "> " : "  ", p->label);
            fb_text(app, 80, py, label, col);
            float m = param_mapped(p);
            if (strcmp(p->unit, "Hz") == 0) snprintf(val, sizeof(val), "%.0fHZ", m);
            else if (strcmp(p->unit, "s") == 0) snprintf(val, sizeof(val), "%.2fS", m);
            else snprintf(val, sizeof(val), "%.2f", p->value);
            fb_text(app, 400, py, val, col);
            fb_rect(app, 650, py + 4, 750, 16, COL_SLIDER);
            int sw = (int)(p->value * 750);
            if (sw > 0) fb_rect(app, 650, py + 4, sw, 16, i == app->selected_param ? COL_GREEN : COL_DIM);
        }

        fb_text(app, 1500, 190, app->reverb_on ? "[REVERB ON]" : "[REVERB]",
                app->reverb_on ? COL_GREEN : COL_DIM);
        char oct[16]; snprintf(oct, sizeof(oct), "OCT: C%d", app->octave);
        fb_text(app, 1500, 240, oct, COL_TEXT);
    }

    /* Keyboard (always visible) */
    draw_keyboard(app, SCREEN_H - 200 - 120);

    /* Help */
    fb_text(app, 80, SCREEN_H - 20, "MIDI  TOUCH TABS  SWIPE SLIDERS  TAP PAGE", COL_DIM);
}

/* ── Audio callback ── */
static void audio_process(App *app, int16_t *out, int frames) {
    float buf808[BLOCK_SIZE], bufkick[BLOCK_SIZE], bufsynth[BLOCK_SIZE], bufsampler[BLOCK_SIZE];
    float mix[BLOCK_SIZE], fx_bus[BLOCK_SIZE];

    pthread_mutex_lock(&app->lock);

    memset(mix, 0, frames * sizeof(float));
    memset(fx_bus, 0, frames * sizeof(float));

    sub808_process(&app->sub808, buf808, frames);
    kick_process(&app->kick, bufkick, frames);
    polysynth_process(&app->synth, bufsynth, frames);
    sampler_process(&app->sampler, bufsampler, frames);

    /* Mix sampler directly (no FX send for now) */
    for (int i = 0; i < frames; i++) mix[i] += bufsampler[i];

    float *bufs[] = {buf808, bufkick, bufsynth};
    for (int e = 0; e < 3; e++) {
        for (int i = 0; i < frames; i++) {
            if (app->reverb_on && app->fx_on[e]) {
                float s = app->fx_send[e];
                fx_bus[i] += bufs[e][i] * s;
                mix[i] += bufs[e][i] * (1.0f - s);
            } else {
                mix[i] += bufs[e][i];
            }
        }
    }

    if (app->reverb_on) {
        float rev_out[BLOCK_SIZE];
        reverb_process(&app->reverb, fx_bus, rev_out, frames);
        for (int i = 0; i < frames; i++) mix[i] += rev_out[i];
    }

    pthread_mutex_unlock(&app->lock);

    /* Peak + limiter */
    float peak = 0;
    for (int i = 0; i < frames; i++) {
        float a = fabsf(mix[i]);
        if (a > peak) peak = a;
    }
    if (peak > 1.0f)
        for (int i = 0; i < frames; i++) mix[i] /= peak;
    app->peak = peak > app->peak ? peak : app->peak * 0.95f;

    /* Interleave stereo S16 */
    for (int i = 0; i < frames; i++) {
        int16_t s = (int16_t)(clampf(mix[i], -1, 1) * 28000.0f);
        out[i * 2] = s;
        out[i * 2 + 1] = s;
    }
}

/* ── Audio thread ── */
static void *audio_thread(void *arg) {
    App *app = (App *)arg;
    snd_pcm_t *pcm;
    int err;

    if ((err = snd_pcm_open(&pcm, "hw:1,0", SND_PCM_STREAM_PLAYBACK, 0)) < 0) {
        fprintf(stderr, "ALSA open: %s\n", snd_strerror(err));
        return NULL;
    }

    snd_pcm_hw_params_t *hw;
    snd_pcm_hw_params_alloca(&hw);
    snd_pcm_hw_params_any(pcm, hw);
    snd_pcm_hw_params_set_access(pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(pcm, hw, SND_PCM_FORMAT_S16_LE);
    unsigned int rate = SR;
    snd_pcm_hw_params_set_rate_near(pcm, hw, &rate, 0);
    snd_pcm_hw_params_set_channels(pcm, hw, 2);
    snd_pcm_uframes_t bsz = BLOCK_SIZE * 4, psz = BLOCK_SIZE;
    snd_pcm_hw_params_set_buffer_size_near(pcm, hw, &bsz);
    snd_pcm_hw_params_set_period_size_near(pcm, hw, &psz, 0);
    snd_pcm_hw_params(pcm, hw);

    printf("Audio: %uHz, buf=%lu, period=%lu\n", rate, bsz, psz);

    int16_t buf[BLOCK_SIZE * 2];
    while (app->running) {
        audio_process(app, buf, BLOCK_SIZE);
        err = snd_pcm_writei(pcm, buf, BLOCK_SIZE);
        if (err == -EPIPE) snd_pcm_prepare(pcm);
        else if (err < 0) { fprintf(stderr, "ALSA: %s\n", snd_strerror(err)); break; }
    }

    snd_pcm_drain(pcm);
    snd_pcm_close(pcm);
    return NULL;
}

/* ── Touch input thread ── */
static void *touch_thread(void *arg) {
    App *app = (App *)arg;
    struct input_event ev;

    app->touch_fd = open("/dev/input/event2", O_RDONLY);
    if (app->touch_fd < 0) { perror("touch open"); return NULL; }

    while (app->running) {
        if (read(app->touch_fd, &ev, sizeof(ev)) != sizeof(ev)) continue;

        if (ev.type == EV_ABS) {
            /* Touch panel is 720x1280, landscape rotated */
            if (ev.code == ABS_X || ev.code == ABS_MT_POSITION_X)
                app->touch_x = ev.value;
            if (ev.code == ABS_Y || ev.code == ABS_MT_POSITION_Y)
                app->touch_y = ev.value;
        }
        if (ev.type == EV_SYN && app->touch_down) {
            int sx = (1280 - app->touch_y) * SCREEN_W / 1280;
            int sy = app->touch_x * SCREEN_H / 720 - 80;
            if (sy >= 170 && sy < 550 && sx >= 650 && sx <= 1400) {
                int pi = (sy - 190) / 90;
                if (pi >= 0 && pi < 4) {
                    float t = clampf((float)(sx - 650) / 750.0f, 0, 1);
                    Param *ap = NULL; int tp = 0;
                    switch (app->engine_idx) {
                        case ENG_808:  ap = app->sub808.params; tp = SUB808_PARAMS; break;
                        case ENG_KICK: ap = app->kick.params; tp = KICK_PARAMS; break;
                        case ENG_SYNTH:ap = app->synth.params; tp = SYNTH_PARAMS; break;
                    }
                    int abs_idx = app->current_page * 4 + pi;
                    if (ap && abs_idx < tp) {
                        pthread_mutex_lock(&app->lock);
                        ap[abs_idx].value = t;
                        pthread_mutex_unlock(&app->lock);
                    }
                }
            }
            /* Slice start/end slider drag (sampler tab) */
            if (app->engine_idx == ENG_SAMPLER && sy >= 700 && sy <= 730) {
                SampleSlot *fsl = &app->sampler.slots[app->sampler.focused_slot];
                if (fsl->loaded && fsl->slice_count > 0) {
                    float sf = (float)fsl->slice_starts[app->selected_slice] / fsl->length;
                    float ef = (float)fsl->slice_ends[app->selected_slice]   / fsl->length;
                    if (sx >= 320 && sx <= 700) {
                        float t = clampf((float)(sx - 320) / 380.0f, 0, 1);
                        pthread_mutex_lock(&app->lock);
                        sampler_set_slice(&app->sampler,
                                          app->sampler.focused_slot,
                                          app->selected_slice, t, ef);
                        pthread_mutex_unlock(&app->lock);
                    } else if (sx >= 1000 && sx <= 1380) {
                        float t = clampf((float)(sx - 1000) / 380.0f, 0, 1);
                        pthread_mutex_lock(&app->lock);
                        sampler_set_slice(&app->sampler,
                                          app->sampler.focused_slot,
                                          app->selected_slice, sf, t);
                        pthread_mutex_unlock(&app->lock);
                    }
                }
            }
        }
        if (ev.type == EV_KEY && ev.code == BTN_TOUCH) {
            if (ev.value == 1 && !app->touch_down) {
                app->touch_down = 1;
                /* Debounce: 300ms between taps */
                struct timespec now;
                clock_gettime(CLOCK_MONOTONIC, &now);
                long ms = (now.tv_sec - app->last_tap.tv_sec) * 1000
                        + (now.tv_nsec - app->last_tap.tv_nsec) / 1000000;
                int debounced = ms < 300;
                app->last_tap = now;
                /* X correct, Y offset corrected — shift up ~150px */
                int sx = (1280 - app->touch_y) * SCREEN_W / 1280;
                int sy = app->touch_x * SCREEN_H / 720 - 80;

                /* Engine tab touch (debounced) */
                if (!debounced && sy >= 50 && sy < 100) {
                    int eng = (sx - 80) / 200;
                    if (eng >= 0 && eng < ENG_COUNT) app->engine_idx = eng;
                }

                /* Keyboard touch */
                if (sy > SCREEN_H - 340) {
                    int key = (sx - 80) / 100;
                    if (key >= 0 && key < 14) {
                        int white_notes[] = {0,2,4,5,7,9,11,12,14,16,17,19,21,23};
                        int note = app->octave * 12 + 24 + white_notes[key % 14];
                        pthread_mutex_lock(&app->lock);
                        switch (app->engine_idx) {
                            case ENG_808:  sub808_trigger(&app->sub808, note, 0.9f); break;
                            case ENG_KICK: kick_trigger(&app->kick, note, 0.9f); break;
                            case ENG_SYNTH:   polysynth_trigger(&app->synth, note, 0.9f); break;
                            case ENG_SAMPLER: sampler_trigger(&app->sampler, note, 0.9f); break;
                        }
                        pthread_mutex_unlock(&app->lock);
                    }
                }

                /* Slider touch — paged */
                if (sy >= 170 && sy < 550 && sx >= 650 && sx <= 1400) {
                    int pi = (sy - 190) / 90;
                    if (pi >= 0 && pi < 4) {
                        float t = clampf((float)(sx - 650) / 750.0f, 0, 1);
                        app->selected_param = pi;
                        Param *ap = NULL; int tp = 0;
                        switch (app->engine_idx) {
                            case ENG_808:  ap = app->sub808.params; tp = SUB808_PARAMS; break;
                            case ENG_KICK: ap = app->kick.params; tp = KICK_PARAMS; break;
                            case ENG_SYNTH:  ap = app->synth.params; tp = SYNTH_PARAMS; break;
                            case ENG_SAMPLER:ap = app->sampler.slots[app->sampler.focused_slot].params; tp = SAMPLER_SLOT_PARAMS; break;
                        }
                        int abs_idx = app->current_page * 4 + pi;
                        if (ap && abs_idx < tp) {
                            pthread_mutex_lock(&app->lock);
                            ap[abs_idx].value = t;
                            pthread_mutex_unlock(&app->lock);
                        }
                    }
                }

                /* Page tap (top-right area near PAGE text) */
                if (!debounced && sx >= 1450 && sx <= 1850 && sy >= 155 && sy <= 195) {
                    int tp = 0;
                    switch (app->engine_idx) {
                        case ENG_808:    tp = SUB808_PARAMS; break;
                        case ENG_KICK:   tp = KICK_PARAMS; break;
                        case ENG_SYNTH:  tp = SYNTH_PARAMS; break;
                        case ENG_SAMPLER:tp = SAMPLER_SLOT_PARAMS; break;
                    }
                    int np = (tp + 3) / 4;
                    if (np > 0) app->current_page = (app->current_page + 1) % np;
                }

                /* Reverb toggle (debounced) */
                if (!debounced && sx >= 1500 && sx <= 1800 && sy >= 170 && sy <= 220) {
                    app->reverb_on = !app->reverb_on;
                    if (!app->reverb_on) reverb_clear(&app->reverb);
                }

                /* SAMPLER tab touch */
                if (app->engine_idx == ENG_SAMPLER) {
                    SampleSlot *fsl = &app->sampler.slots[app->sampler.focused_slot];
                    /* Pad grid tap → focus slot */
                    if (!debounced && sx >= 80 && sx < 520 && sy >= 170 && sy < 510) {
                        int c = (sx - 80) / 110;
                        int r = (sy - 170) / 90;
                        if (c >= 0 && c < 4 && r >= 0 && r < 4) {
                            int idx = r * 4 + c;
                            app->sampler.focused_slot = idx;
                            app->selected_slice = 0;
                        }
                    }
                    /* Slice strip tap → select slice under finger */
                    if (sx >= 80 && sx <= 1440 && sy >= 605 && sy <= 675 &&
                        fsl->slice_count > 0 && fsl->length > 0) {
                        float frac = (float)(sx - 80) / 1360.0f;
                        int sample_pos = (int)(frac * fsl->length);
                        for (int i = 0; i < fsl->slice_count; i++) {
                            if (sample_pos >= fsl->slice_starts[i] &&
                                sample_pos < fsl->slice_ends[i]) {
                                app->selected_slice = i;
                                break;
                            }
                        }
                    }
                    /* Slice nav buttons */
                    if (!debounced && sy >= 565 && sy <= 595) {
                        if (sx >= 700 && sx <= 820 && fsl->slice_count > 0) {
                            app->selected_slice =
                                (app->selected_slice - 1 + fsl->slice_count) % fsl->slice_count;
                        } else if (sx >= 830 && sx <= 950 && fsl->slice_count > 0) {
                            app->selected_slice =
                                (app->selected_slice + 1) % fsl->slice_count;
                        } else if (sx >= 960 && sx <= 1100) {
                            pthread_mutex_lock(&app->lock);
                            sampler_reset_slices(&app->sampler,
                                                 app->sampler.focused_slot);
                            pthread_mutex_unlock(&app->lock);
                            app->selected_slice = 0;
                        }
                    }
                    /* Slice start/end slider tap */
                    if (fsl->loaded && fsl->slice_count > 0 &&
                        sy >= 700 && sy <= 730) {
                        float sf = (float)fsl->slice_starts[app->selected_slice] / fsl->length;
                        float ef = (float)fsl->slice_ends[app->selected_slice]   / fsl->length;
                        if (sx >= 320 && sx <= 700) {
                            float t = clampf((float)(sx - 320) / 380.0f, 0, 1);
                            pthread_mutex_lock(&app->lock);
                            sampler_set_slice(&app->sampler,
                                              app->sampler.focused_slot,
                                              app->selected_slice, t, ef);
                            pthread_mutex_unlock(&app->lock);
                        } else if (sx >= 1000 && sx <= 1380) {
                            float t = clampf((float)(sx - 1000) / 380.0f, 0, 1);
                            pthread_mutex_lock(&app->lock);
                            sampler_set_slice(&app->sampler,
                                              app->sampler.focused_slot,
                                              app->selected_slice, sf, t);
                            pthread_mutex_unlock(&app->lock);
                        }
                    }
                }

                /* CHORD tab touch */
                if (!debounced && app->engine_idx == ENG_CHORD) {
                    /* Progression selection (left panel, y=200..500) */
                    if (sx >= 60 && sx <= 960 && sy >= 200 && sy <= 500) {
                        int pi = (sy - 210) / 48;
                        if (pi >= 0 && pi < NUM_PROGRESSIONS) {
                            app->chord_prog_idx = pi;
                            app->chord_step = 0;
                        }
                    }
                    /* Step selection (right side, y=240..440) */
                    if (sx >= 1000 && sx <= 1400 && sy >= 240 && sy <= 440) {
                        int si = (sy - 240) / 50;
                        int plen = prog_length(app->chord_prog_idx);
                        if (si >= 0 && si < plen) {
                            app->chord_step = si;
                            /* Trigger the chord */
                            int root = app->octave * 12 + 24;
                            int notes[5]; char lbl[16];
                            int count = build_prog_chord(root, app->chord_prog_idx, si, notes, lbl, sizeof(lbl));
                            pthread_mutex_lock(&app->lock);
                            polysynth_release_all(&app->synth);
                            for (int n = 0; n < count; n++)
                                polysynth_trigger(&app->synth, notes[n], 0.8f);
                            pthread_mutex_unlock(&app->lock);
                        }
                    }
                    /* Mode selection (y=500..660) */
                    if (sx >= 1000 && sx <= 1400 && sy >= 500 && sy <= 660) {
                        int mi = (sy - 500) / 40;
                        if (mi >= 0 && mi < 4) app->chord_mode = mi;
                    }
                }
            }
            if (ev.value == 0) {
                app->touch_down = 0;
                /* Release notes on synth/chord */
                if (app->engine_idx == ENG_SYNTH || app->engine_idx == ENG_CHORD) {
                    pthread_mutex_lock(&app->lock);
                    polysynth_release_all(&app->synth);
                    pthread_mutex_unlock(&app->lock);
                }
            }
        }
    }

    close(app->touch_fd);
    return NULL;
}

/* ── MIDI thread ── */
static void *midi_thread(void *arg) {
    App *app = (App *)arg;
    app->midi_fd = open("/dev/midi", O_RDONLY);
    if (app->midi_fd < 0) {
        app->midi_fd = open("/dev/midi2", O_RDONLY);
    }
    if (app->midi_fd < 0) {
        app->midi_fd = open("/dev/snd/midiC0D0", O_RDONLY);
    }
    if (app->midi_fd < 0) {
        fprintf(stderr, "MIDI: no device found\n");
        return NULL;
    }
    fprintf(stderr, "MIDI: connected\n");

    uint8_t buf[3];
    int pos = 0;
    uint8_t status = 0;

    while (app->running) {
        uint8_t byte;
        if (read(app->midi_fd, &byte, 1) != 1) continue;

        /* Running status */
        if (byte & 0x80) {
            status = byte;
            pos = 0;
            continue;
        }

        buf[pos++] = byte;
        int need = 2; /* most messages need 2 data bytes */
        if ((status & 0xF0) == 0xC0 || (status & 0xF0) == 0xD0) need = 1;

        if (pos >= need) {
            int ch = status & 0x0F;
            int cmd = status & 0xF0;
            int note = buf[0] & 0x7F;
            int vel = (need > 1) ? buf[1] & 0x7F : 64;
            pos = 0;

            if (cmd == 0x90 && vel > 0) {
                /* Note On */
                app->active_notes[note] = 1;
                float fvel = vel / 127.0f;
                pthread_mutex_lock(&app->lock);
                if (app->engine_idx == ENG_CHORD) {
                    /* The played note IS the key root — build the chord
                       for the current step around it */
                    int cnotes[5]; char clbl[16];
                    int cnt = build_prog_chord(note, app->chord_prog_idx,
                                               app->chord_step, cnotes, clbl, sizeof(clbl));
                    polysynth_release_all(&app->synth);
                    for (int cn = 0; cn < cnt; cn++)
                        polysynth_trigger(&app->synth, cnotes[cn], fvel);
                    /* Show which notes are active */
                    for (int cn = 0; cn < cnt; cn++)
                        if (cnotes[cn] < 128) app->active_notes[cnotes[cn]] = 1;
                } else {
                    switch (app->engine_idx) {
                        case ENG_808:  sub808_trigger(&app->sub808, note, fvel); break;
                        case ENG_KICK: kick_trigger(&app->kick, note, fvel); break;
                        case ENG_SYNTH:   polysynth_trigger(&app->synth, note, fvel); break;
                            case ENG_SAMPLER: sampler_trigger(&app->sampler, note, fvel); break;
                    }
                }
                pthread_mutex_unlock(&app->lock);
            }
            else if (cmd == 0x80 || (cmd == 0x90 && vel == 0)) {
                /* Note Off */
                app->active_notes[note] = 0;
                pthread_mutex_lock(&app->lock);
                switch (app->engine_idx) {
                    case ENG_808:  sub808_release(&app->sub808, note); break;
                    case ENG_KICK: kick_release(&app->kick, note); break;
                    case ENG_SYNTH:   polysynth_release(&app->synth, note); break;
                    case ENG_SAMPLER: sampler_release(&app->sampler, note); break;
                    case ENG_CHORD:
                        polysynth_release_all(&app->synth);
                        memset(app->active_notes, 0, sizeof(app->active_notes));
                        break;
                }
                pthread_mutex_unlock(&app->lock);
            }
            else if (cmd == 0xB0) {
                /* CC — could map knobs to params later */
            }
        }
    }

    close(app->midi_fd);
    return NULL;
}

/* ── Main ── */
static volatile int quit = 0;
static void sighandler(int s) { quit = 1; }

int main(void) {
    App app;
    memset(&app, 0, sizeof(app));
    pthread_mutex_init(&app.lock, NULL);
    app.running = 1;
    app.octave = 2;
    app.engine_idx = ENG_SYNTH;

    /* Init engines */
    sub808_init(&app.sub808);
    kick_init(&app.kick);
    polysynth_init(&app.synth);
    sampler_init(&app.sampler, SR);
    reverb_init(&app.reverb);

    /* Load any samples in /opt/cypher/samples/ */
    {
        char path[256];
        for (int i = 0; i < 16; i++) {
            snprintf(path, sizeof(path), "/opt/cypher/samples/pad%02d.wav", i);
            FILE *test = fopen(path, "r");
            if (test) { fclose(test); sampler_load_slot(&app.sampler, i, path); }
        }
    }
    app.reverb_on = 0;
    app.fx_on[2] = 1; /* synth send on by default */
    app.fx_send[0] = 0.5f; app.fx_send[1] = 0.5f; app.fx_send[2] = 0.5f;

    /* Open framebuffer */
    app.fb_fd = open("/dev/fb0", O_RDWR);
    if (app.fb_fd < 0) { perror("fb0"); return 1; }
    app.fb = mmap(NULL, FB_STRIDE * FB_H, PROT_READ | PROT_WRITE, MAP_SHARED, app.fb_fd, 0);
    if (app.fb == MAP_FAILED) { perror("mmap"); return 1; }

    signal(SIGINT, sighandler);
    signal(SIGTERM, sighandler);

    /* Start threads */
    pthread_t audio_tid, touch_tid, midi_tid;
    pthread_create(&audio_tid, NULL, audio_thread, &app);
    pthread_create(&touch_tid, NULL, touch_thread, &app);
    pthread_create(&midi_tid, NULL, midi_thread, &app);

    printf("CYPHER running on RK3568\n");
    printf("Touch screen or play MIDI!\n");

    /* UI loop */
    while (!quit && app.running) {
        draw_ui(&app);
        usleep(33333); /* ~30fps */
    }

    app.running = 0;
    pthread_join(audio_tid, NULL);
    pthread_join(touch_tid, NULL);
    pthread_join(midi_tid, NULL);
    munmap(app.fb, FB_STRIDE * FB_H);
    close(app.fb_fd);
    pthread_mutex_destroy(&app.lock);

    printf("CYPHER stopped.\n");
    return 0;
}
