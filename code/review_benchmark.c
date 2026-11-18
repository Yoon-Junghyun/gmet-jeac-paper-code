/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Normalized oxidation rates used for the single-core comparison in section S6.
 *
 * The ts1 branch (method 3) adapts mhc_trapezoid in QSoas src/functions.cc,
 * https://github.com/fourmond/QSoas/blob/f04f01770724316917728abed445049d0006a42d/src/functions.cc
 * The adaptation fixes the step at one, normalizes by the limiting rate
 * and evaluates batches of oxidation rates (October 30, 2026).
 * Names and layout were revised on November 17, 2026. See COPYING for the
 * license accompanying the adapted routine. Upstream notice:
 *
 * Copyright 2015, 2018 by CNRS/AMU
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <math.h>
#include <stddef.h>

#define PI 3.14159265358979323846264338327950288

static double sigmoid(double x) {
    if (x >= 0.0)
        return 1.0 / (1.0 + exp(-x));
    double y = exp(x);
    return y / (1.0 + y);
}

static double erfcx_positive(double x) {
    if (x < 26.0)
        return exp(x * x) * erfc(x);
    double term = 1.0, sum = 1.0;
    double a = 1.0 / (2.0 * x * x);
    for (int k = 1; k <= 12; k++) {
        term *= -(2.0 * k - 1.0) * a;
        sum += term;
    }
    return sum / (sqrt(PI) * x);
}

static double scaled_term(double x, double q) {
    if (x >= 0.0)
        return exp(-q) * erfcx_positive(x);
    return exp(x * x - q) * erfc(x);
}

static double rate(int method, int order, double energy, double eta,
                   const double *nodes, const double *weights) {
    double h = fabs(eta), value = 0.0;
    if (method == 0) {
        const double v[] = {2.0093643274018693, 6.640460404596436};
        const double w[] = {0.7227441690552268, 0.27725583094477313};
        for (int i = 0; i < 2; i++) {
            double d = sqrt(4.0 * energy + 2.0 * v[i]);
            value += 0.5 * w[i] * (erfc((energy - h) / d) + erfc((energy + h) / d));
        }
        value *= sigmoid(h);
    } else if (method == 1) {
        double x = (energy - sqrt(1.0 + sqrt(energy) + h * h)) / (2.0 * sqrt(energy));
        value = 0.5 * erfc(x) * sigmoid(h);
    } else if (method == 2) {
        double q = (energy - h) * (energy - h) / (4.0 * energy);
        double d = 2.0 * sqrt(energy);
        for (int j = 0; j < order; j++) {
            double t = ((2.0 * j + 1.0) * energy + h) / d;
            double p = ((2.0 * j + 1.0) * energy - h) / d;
            value += 0.5 * weights[j] * (scaled_term(t, q) + scaled_term(p, q));
        }
    } else if (method == 3) {
        /* Fourmond and Léger (2020), Eqs. A.3–A.5, evaluated at unit step size.
           QSoas mhc_trapezoid uses the same cutoff and relative stopping rule. */
        double center = energy - h, width = 9.0 * sqrt(energy);
        if (energy > h && width < center + 0.5 * width)
            width = center + 0.5 * width;
        int max_steps = 3 * (int)ceil(width);
        double alpha = exp(-1.0 / (4.0 * energy));
        double alpha_squared = alpha * alpha, factor = alpha;
        double beta = exp(1.0), inverse_beta = 1.0 / beta, gamma = exp(center);
        double positive = beta, negative = inverse_beta, gaussian = 1.0;
        value = 1.0 / (1.0 + gamma);
        for (int i = 1; i <= max_steps; i++) {
            gaussian *= factor;
            factor *= alpha_squared;
            double pair = gaussian / (1.0 + gamma * positive)
                        + gaussian / (1.0 + gamma * negative);
            value += pair;
            if (pair < value * 1e-12)
                break;
            positive *= beta;
            negative *= inverse_beta;
        }
        value /= 2.0 * sqrt(PI * energy);
    } else if (method == 4) {
        double scale = 2.0 * sqrt(energy);
        for (int i = 0; i < order; i++)
            value += weights[i] * sigmoid(h - energy - scale * nodes[i]);
    }
    return value * (eta < 0.0 ? exp(eta) : 1.0);
}

void evaluate_many(int method, int order, size_t n, const double *energy,
                   const double *eta, double *out, const double *nodes,
                   const double *weights) {
    for (size_t i = 0; i < n; i++)
        out[i] = rate(method, order, energy[i], eta[i], nodes, weights);
}
