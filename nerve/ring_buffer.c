/* ring_buffer.c — Lock-free single-producer single-consumer ring buffer.
 *
 * Power-of-2 capacity, atomic head/tail via GCC __atomic builtins.
 * Memory layout: [capacity * elem_size bytes data] at offset 0.
 *
 * Compile: gcc -shared -fPIC -O3 -o nerve/ring_buffer.so nerve/ring_buffer.c
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

/* Cache-line size (assumed 64 bytes; common on x86_64/ARM). */
#define CACHE_LINE 64

/* Pad to cache line to prevent false sharing between head and tail. */
struct rb_head {
    _Atomic uint64_t value;
    char padding[CACHE_LINE - sizeof(_Atomic uint64_t)];
};

struct ring_buffer {
    uint64_t capacity;       /* power of 2 */
    uint64_t elem_size;
    uint64_t mask;
    struct rb_head head;     /* write cursor (producer) */
    struct rb_head tail;     /* read cursor (consumer) */
    char data[];             /* flexible array member */
};

/* Round up to next power of 2. */
static uint64_t next_pow2(uint64_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
}

/* Exported API — C naming so ctypes can find symbols easily. */

void *rb_create(uint64_t capacity, uint64_t elem_size) {
    uint64_t cap = next_pow2(capacity);
    size_t total = sizeof(struct ring_buffer) + cap * elem_size;
    struct ring_buffer *rb = calloc(1, total);
    if (!rb) return NULL;
    rb->capacity = cap;
    rb->elem_size = elem_size;
    rb->mask = cap - 1;
    atomic_store_explicit(&rb->head.value, 0, memory_order_relaxed);
    atomic_store_explicit(&rb->tail.value, 0, memory_order_relaxed);
    return rb;
}

void rb_destroy(void *rb_void) {
    free(rb_void);
}

uint64_t rb_capacity(const void *rb_void) {
    const struct ring_buffer *rb = rb_void;
    return rb->capacity;
}

/* Return number of readable elements. */
uint64_t rb_size(const void *rb_void) {
    const struct ring_buffer *rb = rb_void;
    uint64_t h = atomic_load_explicit(&rb->head.value, memory_order_acquire);
    uint64_t t = atomic_load_explicit(&rb->tail.value, memory_order_acquire);
    return h - t;
}

/* Return free slots. */
uint64_t rb_free(const void *rb_void) {
    const struct ring_buffer *rb = rb_void;
    return rb->capacity - rb_size(rb_void);
}

/* Push one element. Returns 1 on success, 0 if full. */
int rb_push(void *rb_void, const void *elem) {
    struct ring_buffer *rb = rb_void;
    uint64_t h = atomic_load_explicit(&rb->head.value, memory_order_relaxed);
    uint64_t t = atomic_load_explicit(&rb->tail.value, memory_order_acquire);
    if (h - t >= rb->capacity) {
        return 0; /* full */
    }
    size_t offset = (h & rb->mask) * rb->elem_size;
    memcpy(rb->data + offset, elem, rb->elem_size);
    atomic_thread_fence(memory_order_release);
    atomic_store_explicit(&rb->head.value, h + 1, memory_order_release);
    return 1;
}

/* Pop one element into dst. Returns 1 on success, 0 if empty. */
int rb_pop(void *rb_void, void *dst) {
    struct ring_buffer *rb = rb_void;
    uint64_t t = atomic_load_explicit(&rb->tail.value, memory_order_relaxed);
    uint64_t h = atomic_load_explicit(&rb->head.value, memory_order_acquire);
    if (h == t) {
        return 0; /* empty */
    }
    size_t offset = (t & rb->mask) * rb->elem_size;
    memcpy(dst, rb->data + offset, rb->elem_size);
    atomic_thread_fence(memory_order_release);
    atomic_store_explicit(&rb->tail.value, t + 1, memory_order_release);
    return 1;
}

/* Non-blocking peek at front element (copies but doesn't consume). */
int rb_peek(const void *rb_void, void *dst) {
    const struct ring_buffer *rb = rb_void;
    uint64_t t = atomic_load_explicit(&rb->tail.value, memory_order_acquire);
    uint64_t h = atomic_load_explicit(&rb->head.value, memory_order_acquire);
    if (h == t) {
        return 0;
    }
    size_t offset = (t & rb->mask) * rb->elem_size;
    memcpy(dst, rb->data + offset, rb->elem_size);
    return 1;
}

/* Bulk push — copy n elements from src. Returns number actually pushed. */
uint64_t rb_push_bulk(void *rb_void, const void *src, uint64_t n) {
    struct ring_buffer *rb = rb_void;
    uint64_t pushed = 0;
    const char *src_bytes = src;
    while (pushed < n) {
        uint64_t h = atomic_load_explicit(&rb->head.value, memory_order_relaxed);
        uint64_t t = atomic_load_explicit(&rb->tail.value, memory_order_acquire);
        uint64_t free = rb->capacity - (h - t);
        if (free == 0) break;
        uint64_t batch = n - pushed;
        if (batch > free) batch = free;
        /* Copy batch, handling wrap */
        uint64_t idx = h & rb->mask;
        uint64_t to_end = rb->capacity - idx;
        if (batch > to_end) batch = to_end;
        size_t offset = idx * rb->elem_size;
        memcpy(rb->data + offset, src_bytes + pushed * rb->elem_size, batch * rb->elem_size);
        atomic_thread_fence(memory_order_release);
        atomic_store_explicit(&rb->head.value, h + batch, memory_order_release);
        pushed += batch;
    }
    return pushed;
}

/* Bulk pop — copy up to n elements into dst. Returns number popped. */
uint64_t rb_pop_bulk(void *rb_void, void *dst, uint64_t n) {
    struct ring_buffer *rb = rb_void;
    uint64_t popped = 0;
    char *dst_bytes = dst;
    while (popped < n) {
        uint64_t t = atomic_load_explicit(&rb->tail.value, memory_order_relaxed);
        uint64_t h = atomic_load_explicit(&rb->head.value, memory_order_acquire);
        uint64_t avail = h - t;
        if (avail == 0) break;
        uint64_t batch = n - popped;
        if (batch > avail) batch = avail;
        uint64_t idx = t & rb->mask;
        uint64_t to_end = rb->capacity - idx;
        if (batch > to_end) batch = to_end;
        size_t offset = idx * rb->elem_size;
        memcpy(dst_bytes + popped * rb->elem_size, rb->data + offset, batch * rb->elem_size);
        atomic_thread_fence(memory_order_release);
        atomic_store_explicit(&rb->tail.value, t + batch, memory_order_release);
        popped += batch;
    }
    return popped;
}
