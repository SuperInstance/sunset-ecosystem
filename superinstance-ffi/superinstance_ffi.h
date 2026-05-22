#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Norm in the Eisenstein integers: N(a,b) = a² - a·b + b².
 */
int eisenstein_norm(int a, int b);

/**
 * Verify that a set of edges satisfies Laman's condition for a subset.
 *
 * For *k* vertices, at most *2k - 3* edges are allowed for minimal rigidity.
 * Returns 1 if the subset satisfies the condition, 0 otherwise.
 */
int laman_check_subset(unsigned int num_vertices, unsigned int num_edges);

/**
 * Full Laman rigidity test for a graph with n vertices and m edges.
 * Requires m == 2n - 3 for generic minimal rigidity in 2D.
 */
int laman_is_rigid(unsigned int num_vertices, unsigned int num_edges);

/**
 * Check holonomic consistency around a cycle of states.
 *
 * *states* is a flat array of length *len*. The cumulative drift
 * (sum of absolute differences) divided by *len* must be ≤ *threshold*.
 * Returns 1.0 if consistent, 0.0 if not.
 */
float holonomy_check(const double *states, unsigned int len, double threshold);

/**
 * Encode a frequency ratio into Pythagorean 48-tone space.
 *
 * Returns the nearest tempered semitone index (0..47) for a
 * frequency ratio expressed as `numerator / denominator`.
 * The Pythagorean comma (~23.46 cents) is folded into the octave.
 */
int pythagorean48_encode(int numerator, int denominator);

/**
 * Check if *value* lies within [*lower*, *upper*].
 * Returns 1 if satisfied, 0 if violated.
 */
int constraint_check(double value, double lower, double upper);

/**
 * Compute constraint violation distance (0 if satisfied).
 */
double constraint_violation(double value, double lower, double upper);

/**
 * Cubic spline interpolation between two points with tangent control.
 *
 * *t* in [0,1] blends from p0 to p1 using tangents m0 and m1.
 */
double spline_interpolate(double p0, double p1, double m0, double m1, double t);

/**
 * Apply a deadband filter: if |value - last| < deadband, return last.
 * Otherwise return value and update *last via pointer.
 */
double deadband_filter(double value, double *last, double deadband);

/**
 * L1 distance between two float arrays of length *dim*.
 */
float manhattan_distance(const float *a, const float *b, unsigned int dim);

/**
 * Cascade match: compare *query* against *candidates* with tiered thresholds.
 *
 * *candidates* is a flat [n * dim] array. *thresholds* is a [tiers] array
 * of decreasing match thresholds. Returns the index of the first candidate
 * that passes any tier, or -1 if none match.
 */
int cascade_match(const float *query,
                  const float *candidates,
                  unsigned int n,
                  unsigned int dim,
                  const float *thresholds,
                  unsigned int tiers);
