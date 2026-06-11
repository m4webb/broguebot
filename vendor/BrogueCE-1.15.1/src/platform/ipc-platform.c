/* brogue-bot IPC platform: a headless backend for machine-driven play.
 *
 * Protocol: when the game wants input, one binary observation frame is
 * written to fd $BROGUE_IPC_OUT, then one little-endian uint16 keycode is
 * read from fd $BROGUE_IPC_IN. No terminal, no rendering, no delays.
 *
 * Fairness contract: every field in the frame is information a human
 * player can see on screen. The map is the game's own display buffer
 * (post field-of-view, post map-memory). Unidentified items expose no
 * true kind; HP/nutrition are quantized to the sidebar bar's resolution.
 *
 * Frame layout (packed, little-endian), version 1:
 *   header   : u32 magic=0x42424631 ("BBF1"), u8 type (0=obs 1=done),
 *              u8 version, u16 pad, u32 seq
 *   stats    : see bb_stats below (40 bytes)
 *   monsters : u8 count, then 24 x bb_monster (8 bytes each)
 *   items    : u8 count, then 26 x bb_item (48 bytes each)
 *   messages : 4 x { u8 len, char text[99] } (oldest first)
 *   map      : COLS*ROWS cells x { u16 glyph, u8 fg[3], u8 bg[3] }
 *              column-major (x outer, y inner), colors on 0-100 scale
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <poll.h>
#include <errno.h>

#include "platform.h"
#include "GlobalsBase.h"
#include "Globals.h"

#define BB_MAGIC      0x42424631u
#define BB_VERSION    1
#define BB_TYPE_OBS   0
#define BB_TYPE_DONE  1
#define BB_MAX_MONSTERS 24
#define BB_MAX_ITEMS    26
#define BB_NUM_MESSAGES 4
#define BB_MSG_LEN      99
#define BB_BAR_CELLS    20   /* sidebar bars are 20 cells wide */

#pragma pack(push, 1)
typedef struct bb_header {
    unsigned int magic;
    unsigned char type;
    unsigned char version;
    unsigned short pad;
    unsigned int seq;
} bb_header;

typedef struct bb_stats {
    unsigned short depth;
    unsigned short deepest;
    unsigned short hp_q;          /* 0..BB_BAR_CELLS */
    unsigned short nutrition_q;   /* 0..BB_BAR_CELLS */
    short strength;               /* as displayed (weakness applied) */
    short armor;                  /* displayedArmorValue() */
    unsigned int gold;
    unsigned int player_turns;
    unsigned int absolute_turns;
    unsigned int status_mask;     /* bit i = statusCondition i active+named */
    unsigned char px, py;
    unsigned char game_has_ended;
    unsigned char weapon_letter;  /* 0 if none equipped */
    unsigned char armor_letter;   /* 0 if none equipped */
    unsigned char pad[3];
    unsigned short stealth_range;
    unsigned short pad2;
} bb_stats;

typedef struct bb_monster {
    unsigned short glyph;
    unsigned char x, y;
    unsigned char hp_q;           /* 0..BB_BAR_CELLS */
    unsigned char state;          /* enum creatureStates */
    unsigned char flags;          /* 1=captive */
    unsigned char pad;
} bb_monster;

typedef struct bb_item {
    unsigned char letter;
    unsigned char category_bit;   /* ffs(category)-1 */
    unsigned char kind;           /* 255 when not player-known */
    unsigned char flags;          /* 1=equipped 2=identified */
    signed char enchant;          /* -128 when not player-known */
    unsigned char quantity;
    signed char str_req;
    unsigned char pad;
    char name[40];                /* itemName(), the player-visible string */
} bb_item;

typedef struct bb_message {
    unsigned char len;
    char text[BB_MSG_LEN];
} bb_message;

typedef struct bb_cell {
    unsigned short glyph;
    unsigned char fg[3];
    unsigned char bg[3];
} bb_cell;

typedef struct bb_frame {
    bb_header header;
    bb_stats stats;
    unsigned char monster_count;
    bb_monster monsters[BB_MAX_MONSTERS];
    unsigned char item_count;
    bb_item items[BB_MAX_ITEMS];
    bb_message messages[BB_NUM_MESSAGES];
    bb_cell map[COLS][ROWS];
} bb_frame;
#pragma pack(pop)

static int bb_outFd = -1;
static int bb_inFd = -1;
static unsigned int bb_seq = 0;
static bb_frame bb_buf;
static int bb_gameOver = 0;  /* set by notifyEvent; consumed at next input */

/* ------------------------------------------------------------ helpers */

static void bb_writeAll(const void *data, size_t len) {
    const char *p = data;
    while (len > 0) {
        ssize_t n = write(bb_outFd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            exit(70);  /* driver hung up: no reason to keep playing */
        }
        p += n;
        len -= (size_t) n;
    }
}

static boolean bb_readKey(unsigned short *key) {
    unsigned char buf[2];
    size_t got = 0;
    while (got < 2) {
        ssize_t n = read(bb_inFd, buf + got, 2 - got);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) exit(70);  /* driver hung up */
        got += (size_t) n;
    }
    *key = (unsigned short) (buf[0] | (buf[1] << 8));
    return true;
}

static unsigned char bb_quantizeBar(long cur, long max) {
    if (max <= 0) return 0;
    if (cur <= 0) return 0;
    long q = (cur * BB_BAR_CELLS + max - 1) / max;  /* ceil: alive shows >=1 */
    if (q > BB_BAR_CELLS) q = BB_BAR_CELLS;
    return (unsigned char) q;
}

/* The kind (true identity) of flavored categories is player-known only
   once the item or its whole kind has been identified. Other categories
   (weapons, armor, food...) always display their kind. */
static boolean bb_kindKnown(const item *theItem) {
    const itemTable *table = NULL;
    switch (theItem->category) {
        case POTION: table = potionTable; break;
        case SCROLL: table = scrollTable; break;
        case WAND:   table = wandTable;   break;
        case STAFF:  table = staffTable;  break;
        case RING:   table = ringTable;   break;
        default: return true;
    }
    if (theItem->flags & ITEM_IDENTIFIED) return true;
    return table[theItem->kind].identified;
}

static unsigned char bb_categoryBit(unsigned short category) {
    unsigned char bit = 0;
    while (category > 1) {
        category >>= 1;
        bit++;
    }
    return bit;
}

/* ------------------------------------------------------------ frame */

static void bb_fillStats(bb_stats *s) {
    memset(s, 0, sizeof(*s));
    s->depth = (unsigned short) rogue.depthLevel;
    s->deepest = (unsigned short) rogue.deepestLevel;
    s->hp_q = bb_quantizeBar(player.currentHP, player.info.maxHP);
    s->nutrition_q = bb_quantizeBar(player.status[STATUS_NUTRITION], STOMACH_SIZE);
    s->strength = (short) (rogue.strength - player.weaknessAmount);
    s->armor = displayedArmorValue();
    s->gold = (unsigned int) rogue.gold;
    s->player_turns = (unsigned int) rogue.playerTurnNumber;
    s->absolute_turns = (unsigned int) rogue.absoluteTurnNumber;
    s->px = (unsigned char) player.loc.x;
    s->py = (unsigned char) player.loc.y;
    s->game_has_ended = rogue.gameHasEnded ? 1 : 0;
    s->stealth_range = (unsigned short) rogue.stealthRange;
    s->weapon_letter = rogue.weapon ? (unsigned char) rogue.weapon->inventoryLetter : 0;
    s->armor_letter = rogue.armor ? (unsigned char) rogue.armor->inventoryLetter : 0;
    for (int i = 0; i < NUMBER_OF_STATUS_EFFECTS && i < 32; i++) {
        if (statusEffectCatalog[i].name[0] && player.status[i] > 0) {
            s->status_mask |= (1u << i);
        }
    }
}

static unsigned char bb_fillMonsters(bb_monster *out) {
    unsigned char count = 0;
    for (creatureIterator it = iterateCreatures(monsters);
            hasNextCreature(it) && count < BB_MAX_MONSTERS;) {
        creature *monst = nextCreature(&it);
        if (!canSeeMonster(monst)) continue;
        bb_monster *m = &out[count++];
        m->glyph = (unsigned short) monst->info.displayChar;
        m->x = (unsigned char) monst->loc.x;
        m->y = (unsigned char) monst->loc.y;
        m->hp_q = bb_quantizeBar(monst->currentHP, monst->info.maxHP);
        m->state = (unsigned char) monst->creatureState;
        m->flags = (monst->bookkeepingFlags & MB_CAPTIVE) ? 1 : 0;
        m->pad = 0;
    }
    return count;
}

static unsigned char bb_fillItems(bb_item *out) {
    unsigned char count = 0;
    char nameBuf[COLS * 3];
    for (item *theItem = packItems->nextItem;
            theItem != NULL && count < BB_MAX_ITEMS;
            theItem = theItem->nextItem) {
        bb_item *r = &out[count++];
        memset(r, 0, sizeof(*r));
        boolean known = bb_kindKnown(theItem);
        r->letter = (unsigned char) theItem->inventoryLetter;
        r->category_bit = bb_categoryBit(theItem->category);
        r->kind = known ? (unsigned char) theItem->kind : 255;
        r->flags = ((theItem->flags & ITEM_EQUIPPED) ? 1 : 0)
                 | ((theItem->flags & ITEM_IDENTIFIED) ? 2 : 0);
        r->enchant = (theItem->flags & ITEM_IDENTIFIED)
                 ? (signed char) theItem->enchant1 : -128;
        r->quantity = (unsigned char) (theItem->quantity > 255
                 ? 255 : theItem->quantity);
        r->str_req = (signed char) theItem->strengthRequired;
        /* itemName prints exactly what the inventory screen shows, so it
           already respects identification state */
        nameBuf[0] = '\0';
        itemName(theItem, nameBuf, true, false, NULL);
        strncpy(r->name, nameBuf, sizeof(r->name) - 1);
    }
    return count;
}

static void bb_fillMessages(bb_message *out) {
    for (int i = 0; i < BB_NUM_MESSAGES; i++) {
        /* oldest of the window first; back index 0 = most recent */
        int back = BB_NUM_MESSAGES - 1 - i;
        const archivedMessage *m = &messageArchive[
            (messageArchivePosition + MESSAGE_ARCHIVE_ENTRIES - back - 1)
            % MESSAGE_ARCHIVE_ENTRIES];
        /* copy, skipping brogue's inline color escapes (4 bytes each) */
        size_t len = 0;
        for (const char *p = m->message; *p && len < BB_MSG_LEN; p++) {
            if (*p == COLOR_ESCAPE) {
                if (p[1] && p[2] && p[3]) p += 3;
                continue;
            }
            out[i].text[len++] = *p;
        }
        out[i].len = (unsigned char) len;
        memset(out[i].text + len, 0, BB_MSG_LEN - len);
    }
}

static void bb_emitFrame(unsigned char type) {
    bb_frame *f = &bb_buf;
    f->header.magic = BB_MAGIC;
    f->header.type = type;
    f->header.version = BB_VERSION;
    f->header.pad = 0;
    f->header.seq = bb_seq++;
    bb_fillStats(&f->stats);
    f->monster_count = bb_fillMonsters(f->monsters);
    f->item_count = bb_fillItems(f->items);
    bb_fillMessages(f->messages);
    for (int x = 0; x < COLS; x++) {
        for (int y = 0; y < ROWS; y++) {
            const cellDisplayBuffer *c = &displayBuffer.cells[x][y];
            bb_cell *o = &f->map[x][y];
            o->glyph = (unsigned short) c->character;
            o->fg[0] = (unsigned char) c->foreColorComponents[0];
            o->fg[1] = (unsigned char) c->foreColorComponents[1];
            o->fg[2] = (unsigned char) c->foreColorComponents[2];
            o->bg[0] = (unsigned char) c->backColorComponents[0];
            o->bg[1] = (unsigned char) c->backColorComponents[1];
            o->bg[2] = (unsigned char) c->backColorComponents[2];
        }
    }
    bb_writeAll(f, sizeof(*f));
}

/* ------------------------------------------------------------ hooks */

static void ipc_gameLoop(void) {
    const char *outEnv = getenv("BROGUE_IPC_OUT");
    const char *inEnv = getenv("BROGUE_IPC_IN");
    if (!outEnv || !inEnv) {
        fprintf(stderr, "BROGUE_IPC_OUT/BROGUE_IPC_IN must be fd numbers\n");
        exit(64);
    }
    bb_outFd = atoi(outEnv);
    bb_inFd = atoi(inEnv);
    /* skip interactive high-score/save-recording screens at game over */
    serverMode = true;
    int status = rogueMain();
    bb_emitFrame(BB_TYPE_DONE);
    exit(status);
}

/* One process == one episode. rogueMain never returns with --no-menu (it
   loops into the main menu / a fresh game), so end the process at the
   first input request after the game-over notification — by then the
   recording and the run history have both been written. */
static void bb_exitIfGameOver(void) {
    if (bb_gameOver) {
        bb_emitFrame(BB_TYPE_DONE);
        exit(0);
    }
}

static boolean ipc_pauseForMilliseconds(short milliseconds, PauseBehavior behavior) {
    (void) milliseconds; (void) behavior;
    bb_exitIfGameOver();
    struct pollfd pfd = { .fd = bb_inFd, .events = POLLIN };
    return poll(&pfd, 1, 0) > 0;  /* never sleep; report pending input */
}

static void ipc_nextKeyOrMouseEvent(rogueEvent *returnEvent, boolean textInput, boolean colorsDance) {
    (void) textInput; (void) colorsDance;
    unsigned short key;
    bb_exitIfGameOver();
    bb_emitFrame(BB_TYPE_OBS);
    bb_readKey(&key);
    returnEvent->eventType = KEYSTROKE;
    returnEvent->param1 = key;
    returnEvent->param2 = 0;
    returnEvent->controlKey = false;
    returnEvent->shiftKey = (key >= 'A' && key <= 'Z');
}

static void ipc_plotChar(enum displayGlyph inputChar,
                         short x, short y,
                         short foreRed, short foreGreen, short foreBlue,
                         short backRed, short backGreen, short backBlue) {
    /* the frame reads displayBuffer directly; nothing to do per cell */
    (void) inputChar; (void) x; (void) y;
    (void) foreRed; (void) foreGreen; (void) foreBlue;
    (void) backRed; (void) backGreen; (void) backBlue;
}

static void ipc_remap(const char *input, const char *output) {
    (void) input; (void) output;
}

static boolean ipc_modifierHeld(int modifier) {
    (void) modifier;
    return false;
}

static void ipc_notifyEvent(short eventId, int data1, int data2,
                            const char *str1, const char *str2) {
    (void) data1; (void) data2; (void) str1; (void) str2;
    if (eventId == GAMEOVER_DEATH || eventId == GAMEOVER_QUIT
            || eventId == GAMEOVER_VICTORY
            || eventId == GAMEOVER_SUPERVICTORY) {
        bb_gameOver = 1;
    }
}

struct brogueConsole ipcConsole = {
    ipc_gameLoop,
    ipc_pauseForMilliseconds,
    ipc_nextKeyOrMouseEvent,
    ipc_plotChar,
    ipc_remap,
    ipc_modifierHeld,
    ipc_notifyEvent,
    NULL,  /* takeScreenshot */
    NULL,  /* setGraphicsMode */
};
