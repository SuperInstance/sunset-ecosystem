/* bloom_filter.c — Bit-packed Bloom filter with multi-hash.
 *
 * Compile: gcc -shared -fPIC -O3 -o nerve/bloom_filter.so nerve/bloom_filter.c
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Murmur-like hash mixing. */
static uint64_t hash_mix(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

/* FNV-1a 64-bit. */
static uint64_t fnv1a(const void *data, size_t len) {
    const uint8_t *bytes = data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= bytes[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

struct bloom_filter {
    uint64_t num_bits;
    uint64_t num_hashes;
    uint64_t num_items;
    uint8_t bits[];  /* flexible array */
};

/* Optimal bits = -(n * ln(p)) / (ln(2)^2); hashes = (m/n) * ln(2). */
void *bf_create(uint64_t expected_items, double false_positive_rate) {
    if (expected_items == 0) expected_items = 1000;
    if (false_positive_rate <= 0 || false_positive_rate >= 1) false_positive_rate = 0.01;

    double ln2 = 0.6931471805599453;
    double m = -(double)expected_items * log(false_positive_rate) / (ln2 * ln2);
    uint64_t num_bits = (uint64_t)(m + 0.5);
    if (num_bits < 64) num_bits = 64;
    /* Round up to multiple of 8 for byte alignment */
    num_bits = (num_bits + 7) & ~7ULL;

    uint64_t num_hashes = (uint64_t)((num_bits / (double)expected_items) * ln2 + 0.5);
    if (num_hashes < 1) num_hashes = 1;
    if (num_hashes > 16) num_hashes = 16;

    size_t total = sizeof(struct bloom_filter) + (num_bits / 8);
    struct bloom_filter *bf = calloc(1, total);
    if (!bf) return NULL;
    bf->num_bits = num_bits;
    bf->num_hashes = num_hashes;
    bf->num_items = 0;
    return bf;
}

void bf_destroy(void *bf_void) {
    free(bf_void);
}

uint64_t bf_num_bits(const void *bf_void) {
    return ((const struct bloom_filter *)bf_void)->num_bits;
}
uint64_t bf_num_hashes(const void *bf_void) {
    return ((const struct bloom_filter *)bf_void)->num_hashes;
}
uint64_t bf_num_items(const void *bf_void) {
    return ((const struct bloom_filter *)bf_void)->num_items;
}

static inline void set_bit(struct bloom_filter *bf, uint64_t idx) {
    bf->bits[idx >> 3] |= (1ULL << (idx & 7));
}

static inline int test_bit(const struct bloom_filter *bf, uint64_t idx) {
    return (bf->bits[idx >> 3] >> (idx & 7)) & 1;
}

void bf_add(void *bf_void, const void *data, size_t len) {
    struct bloom_filter *bf = bf_void;
    uint64_t h1 = fnv1a(data, len);
    uint64_t h2 = hash_mix(h1);
    for (uint64_t i = 0; i < bf->num_hashes; i++) {
        uint64_t idx = (h1 + i * h2) % bf->num_bits;
        set_bit(bf, idx);
    }
    bf->num_items++;
}

int bf_test(const void *bf_void, const void *data, size_t len) {
    const struct bloom_filter *bf = bf_void;
    uint64_t h1 = fnv1a(data, len);
    uint64_t h2 = hash_mix(h1);
    for (uint64_t i = 0; i < bf->num_hashes; i++) {
        uint64_t idx = (h1 + i * h2) % bf->num_bits;
        if (!test_bit(bf, idx)) {
            return 0; /* definitely not present */
        }
    }
    return 1; /* probably present */
}

void bf_clear(void *bf_void) {
    struct bloom_filter *bf = bf_void;
    size_t byte_size = bf->num_bits / 8;
    memset(bf->bits, 0, byte_size);
    bf->num_items = 0;
}
