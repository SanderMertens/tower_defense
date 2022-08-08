// Comment out this line when using as DLL
#define cglm_STATIC
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_h
#define cglm_h

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_common_h
#define cglm_common_h

#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES       /* for windows */
#endif

#ifndef _CRT_SECURE_NO_WARNINGS
#  define _CRT_SECURE_NO_WARNINGS /* for windows */
#endif

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>

#if defined(_MSC_VER)
#  ifdef CGLM_STATIC
#    define CGLM_EXPORT
#  elif defined(CGLM_EXPORTS)
#    define CGLM_EXPORT __declspec(dllexport)
#  else
#    define CGLM_EXPORT __declspec(dllimport)
#  endif
#  define CGLM_INLINE __forceinline
#else
#  define CGLM_EXPORT __attribute__((visibility("default")))
#  define CGLM_INLINE static inline __attribute((always_inline))
#endif

#define GLM_SHUFFLE4(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))
#define GLM_SHUFFLE3(z, y, x)    (((z) << 4) | ((y) << 2) | (x))

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_types_h
#define cglm_types_h

#if defined(_MSC_VER)
/* do not use alignment for older visual studio versions */
#  if _MSC_VER < 1913 /*  Visual Studio 2017 version 15.6  */
#    define CGLM_ALL_UNALIGNED
#    define CGLM_ALIGN(X) /* no alignment */
#  else
#    define CGLM_ALIGN(X) __declspec(align(X))
#  endif
#else
#  define CGLM_ALIGN(X) __attribute((aligned(X)))
#endif

#ifndef CGLM_ALL_UNALIGNED
#  define CGLM_ALIGN_IF(X) CGLM_ALIGN(X)
#else
#  define CGLM_ALIGN_IF(X) /* no alignment */
#endif

#ifdef __AVX__
#  define CGLM_ALIGN_MAT CGLM_ALIGN(32)
#else
#  define CGLM_ALIGN_MAT CGLM_ALIGN(16)
#endif

#ifdef __GNUC__
#  define CGLM_ASSUME_ALIGNED(expr, alignment) \
  __builtin_assume_aligned((expr), (alignment))
#else
#  define CGLM_ASSUME_ALIGNED(expr, alignment) (expr)
#endif

#define CGLM_CASTPTR_ASSUME_ALIGNED(expr, type) \
  ((type*)CGLM_ASSUME_ALIGNED((expr), __alignof__(type)))

typedef float                   vec2[2];
typedef float                   vec3[3];
typedef int                    ivec3[3];
typedef CGLM_ALIGN_IF(16) float vec4[4];
typedef vec4                    versor;     /* |x, y, z, w| -> w is the last */
typedef vec3                    mat3[3];
typedef CGLM_ALIGN_IF(16) vec2  mat2[2];
typedef CGLM_ALIGN_MAT    vec4  mat4[4];

/*
  Important: cglm stores quaternion as [x, y, z, w] in memory since v0.4.0 
  it was [w, x, y, z] before v0.4.0 ( v0.3.5 and earlier ). w is real part.
*/

#define GLM_E         2.71828182845904523536028747135266250   /* e           */
#define GLM_LOG2E     1.44269504088896340735992468100189214   /* log2(e)     */
#define GLM_LOG10E    0.434294481903251827651128918916605082  /* log10(e)    */
#define GLM_LN2       0.693147180559945309417232121458176568  /* loge(2)     */
#define GLM_LN10      2.30258509299404568401799145468436421   /* loge(10)    */
#define GLM_PI        3.14159265358979323846264338327950288   /* pi          */
#define GLM_PI_2      1.57079632679489661923132169163975144   /* pi/2        */
#define GLM_PI_4      0.785398163397448309615660845819875721  /* pi/4        */
#define GLM_1_PI      0.318309886183790671537767526745028724  /* 1/pi        */
#define GLM_2_PI      0.636619772367581343075535053490057448  /* 2/pi        */
#define GLM_2_SQRTPI  1.12837916709551257389615890312154517   /* 2/sqrt(pi)  */
#define GLM_SQRT2     1.41421356237309504880168872420969808   /* sqrt(2)     */
#define GLM_SQRT1_2   0.707106781186547524400844362104849039  /* 1/sqrt(2)   */

#define GLM_Ef        ((float)GLM_E)
#define GLM_LOG2Ef    ((float)GLM_LOG2E)
#define GLM_LOG10Ef   ((float)GLM_LOG10E)
#define GLM_LN2f      ((float)GLM_LN2)
#define GLM_LN10f     ((float)GLM_LN10)
#define GLM_PIf       ((float)GLM_PI)
#define GLM_PI_2f     ((float)GLM_PI_2)
#define GLM_PI_4f     ((float)GLM_PI_4)
#define GLM_1_PIf     ((float)GLM_1_PI)
#define GLM_2_PIf     ((float)GLM_2_PI)
#define GLM_2_SQRTPIf ((float)GLM_2_SQRTPI)
#define GLM_SQRT2f    ((float)GLM_SQRT2)
#define GLM_SQRT1_2f  ((float)GLM_SQRT1_2)

/* DEPRECATED! use GLM_PI and friends */
#define CGLM_PI       GLM_PIf
#define CGLM_PI_2     GLM_PI_2f
#define CGLM_PI_4     GLM_PI_4f

#endif /* cglm_types_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_intrin_h
#define cglm_intrin_h

#if defined( _MSC_VER )
#  if (defined(_M_AMD64) || defined(_M_X64)) || _M_IX86_FP == 2
#    ifndef __SSE2__
#      define __SSE2__
#    endif
#  elif _M_IX86_FP == 1
#    ifndef __SSE__
#      define __SSE__
#    endif
#  endif
/* do not use alignment for older visual studio versions */
#  if _MSC_VER < 1913     /* Visual Studio 2017 version 15.6 */
#    define CGLM_ALL_UNALIGNED
#  endif
#endif

#if defined( __SSE__ ) || defined( __SSE2__ )
#  include <xmmintrin.h>
#  include <emmintrin.h>
#  define CGLM_SSE_FP 1
#  ifndef CGLM_SIMD_x86
#    define CGLM_SIMD_x86
#  endif
#endif

#if defined(__SSE3__)
#  include <x86intrin.h>
#  ifndef CGLM_SIMD_x86
#    define CGLM_SIMD_x86
#  endif
#endif

#if defined(__SSE4_1__)
#  include <smmintrin.h>
#  ifndef CGLM_SIMD_x86
#    define CGLM_SIMD_x86
#  endif
#endif

#if defined(__SSE4_2__)
#  include <nmmintrin.h>
#  ifndef CGLM_SIMD_x86
#    define CGLM_SIMD_x86
#  endif
#endif

#ifdef __AVX__
#  include <immintrin.h>
#  define CGLM_AVX_FP 1
#  ifndef CGLM_SIMD_x86
#    define CGLM_SIMD_x86
#  endif
#endif

/* ARM Neon */
#if defined(__ARM_NEON)
#  include <arm_neon.h>
#  if defined(__ARM_NEON_FP)
#    define CGLM_NEON_FP 1
#    ifndef CGLM_SIMD_ARM
#      define CGLM_SIMD_ARM
#    endif
#  endif
#endif

#if defined(CGLM_SIMD_x86) || defined(CGLM_NEON_FP)
#  ifndef CGLM_SIMD
#    define CGLM_SIMD
#  endif
#endif

#if defined(CGLM_SIMD_x86)
/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_simd_x86_h
#define cglm_simd_x86_h
#ifdef CGLM_SIMD_x86

#ifdef CGLM_ALL_UNALIGNED
#  define glmm_load(p)      _mm_loadu_ps(p)
#  define glmm_store(p, a)  _mm_storeu_ps(p, a)
#else
#  define glmm_load(p)      _mm_load_ps(p)
#  define glmm_store(p, a)  _mm_store_ps(p, a)
#endif

#ifdef CGLM_USE_INT_DOMAIN
#  define glmm_shuff1(xmm, z, y, x, w)                                        \
     _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(xmm),                \
                                        _MM_SHUFFLE(z, y, x, w)))
#else
#  define glmm_shuff1(xmm, z, y, x, w)                                        \
       _mm_shuffle_ps(xmm, xmm, _MM_SHUFFLE(z, y, x, w))
#endif

#define glmm_shuff1x(xmm, x) glmm_shuff1(xmm, x, x, x, x)
#define glmm_shuff2(a, b, z0, y0, x0, w0, z1, y1, x1, w1)                     \
     glmm_shuff1(_mm_shuffle_ps(a, b, _MM_SHUFFLE(z0, y0, x0, w0)),           \
                 z1, y1, x1, w1)

#ifdef __AVX__
#  ifdef CGLM_ALL_UNALIGNED
#    define glmm_load256(p)      _mm256_loadu_ps(p)
#    define glmm_store256(p, a)  _mm256_storeu_ps(p, a)
#  else
#    define glmm_load256(p)      _mm256_load_ps(p)
#    define glmm_store256(p, a)  _mm256_store_ps(p, a)
#  endif
#endif

static inline
__m128
glmm_abs(__m128 x) {
  return _mm_andnot_ps(_mm_set1_ps(-0.0f), x);
}

static inline
__m128
glmm_vhadds(__m128 v) {
#if defined(__SSE3__)
  __m128 shuf, sums;
  shuf = _mm_movehdup_ps(v);
  sums = _mm_add_ps(v, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return sums;
#else
  __m128 shuf, sums;
  shuf = glmm_shuff1(v, 2, 3, 0, 1);
  sums = _mm_add_ps(v, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return sums;
#endif
}

static inline
float
glmm_hadd(__m128 v) {
  return _mm_cvtss_f32(glmm_vhadds(v));
}

static inline
__m128
glmm_vhmin(__m128 v) {
  __m128 x0, x1, x2;
  x0 = _mm_movehl_ps(v, v);     /* [2, 3, 2, 3] */
  x1 = _mm_min_ps(x0, v);       /* [0|2, 1|3, 2|2, 3|3] */
  x2 = glmm_shuff1x(x1, 1);     /* [1|3, 1|3, 1|3, 1|3] */
  return _mm_min_ss(x1, x2);
}

static inline
float
glmm_hmin(__m128 v) {
  return _mm_cvtss_f32(glmm_vhmin(v));
}

static inline
__m128
glmm_vhmax(__m128 v) {
  __m128 x0, x1, x2;
  x0 = _mm_movehl_ps(v, v);     /* [2, 3, 2, 3] */
  x1 = _mm_max_ps(x0, v);       /* [0|2, 1|3, 2|2, 3|3] */
  x2 = glmm_shuff1x(x1, 1);     /* [1|3, 1|3, 1|3, 1|3] */
  return _mm_max_ss(x1, x2);
}

static inline
float
glmm_hmax(__m128 v) {
  return _mm_cvtss_f32(glmm_vhmax(v));
}

static inline
__m128
glmm_vdots(__m128 a, __m128 b) {
#if (defined(__SSE4_1__) || defined(__SSE4_2__)) && defined(CGLM_SSE4_DOT)
  return _mm_dp_ps(a, b, 0xFF);
#elif defined(__SSE3__) && defined(CGLM_SSE3_DOT)
  __m128 x0, x1;
  x0 = _mm_mul_ps(a, b);
  x1 = _mm_hadd_ps(x0, x0);
  return _mm_hadd_ps(x1, x1);
#else
  return glmm_vhadds(_mm_mul_ps(a, b));
#endif
}

static inline
__m128
glmm_vdot(__m128 a, __m128 b) {
#if (defined(__SSE4_1__) || defined(__SSE4_2__)) && defined(CGLM_SSE4_DOT)
  return _mm_dp_ps(a, b, 0xFF);
#elif defined(__SSE3__) && defined(CGLM_SSE3_DOT)
  __m128 x0, x1;
  x0 = _mm_mul_ps(a, b);
  x1 = _mm_hadd_ps(x0, x0);
  return _mm_hadd_ps(x1, x1);
#else
  __m128 x0;
  x0 = _mm_mul_ps(a, b);
  x0 = _mm_add_ps(x0, glmm_shuff1(x0, 1, 0, 3, 2));
  return _mm_add_ps(x0, glmm_shuff1(x0, 0, 1, 0, 1));
#endif
}

static inline
float
glmm_dot(__m128 a, __m128 b) {
  return _mm_cvtss_f32(glmm_vdots(a, b));
}

static inline
float
glmm_norm(__m128 a) {
  return _mm_cvtss_f32(_mm_sqrt_ss(glmm_vhadds(_mm_mul_ps(a, a))));
}

static inline
float
glmm_norm2(__m128 a) {
  return _mm_cvtss_f32(glmm_vhadds(_mm_mul_ps(a, a)));
}

static inline
float
glmm_norm_one(__m128 a) {
  return _mm_cvtss_f32(glmm_vhadds(glmm_abs(a)));
}

static inline
float
glmm_norm_inf(__m128 a) {
  return _mm_cvtss_f32(glmm_vhmax(glmm_abs(a)));
}

static inline
__m128
glmm_load3(float v[3]) {
  __m128i xy;
  __m128  z;

  xy = _mm_loadl_epi64(CGLM_CASTPTR_ASSUME_ALIGNED(v, const __m128i));
  z  = _mm_load_ss(&v[2]);

  return _mm_movelh_ps(_mm_castsi128_ps(xy), z);
}

static inline
void
glmm_store3(float v[3], __m128 vx) {
  _mm_storel_pi(CGLM_CASTPTR_ASSUME_ALIGNED(v, __m64), vx);
  _mm_store_ss(&v[2], glmm_shuff1(vx, 2, 2, 2, 2));
}

#endif
#endif /* cglm_simd_x86_h */

#endif

#if defined(CGLM_SIMD_ARM)
/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_simd_arm_h
#define cglm_simd_arm_h
#ifdef CGLM_SIMD_ARM

#define glmm_load(p)      vld1q_f32(p)
#define glmm_store(p, a)  vst1q_f32(p, a)

static inline
float32x4_t
glmm_abs(float32x4_t v) {
  return vabsq_f32(v);
}

static inline
float
glmm_hadd(float32x4_t v) {
#if defined(__aarch64__)
  return vaddvq_f32(v);
#else
  v = vaddq_f32(v, vrev64q_f32(v));
  v = vaddq_f32(v, vcombine_f32(vget_high_f32(v), vget_low_f32(v)));
  return vgetq_lane_f32(v, 0);
#endif
}

static inline
float
glmm_hmin(float32x4_t v) {
  float32x2_t t;
  t = vpmin_f32(vget_low_f32(v), vget_high_f32(v));
  t = vpmin_f32(t, t);
  return vget_lane_f32(t, 0);
}

static inline
float
glmm_hmax(float32x4_t v) {
  float32x2_t t;
  t = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
  t = vpmax_f32(t, t);
  return vget_lane_f32(t, 0);
}

static inline
float
glmm_dot(float32x4_t a, float32x4_t b) {
  return glmm_hadd(vmulq_f32(a, b));
}

static inline
float
glmm_norm(float32x4_t a) {
  return sqrtf(glmm_dot(a, a));
}

static inline
float
glmm_norm2(float32x4_t a) {
  return glmm_dot(a, a);
}

static inline
float
glmm_norm_one(float32x4_t a) {
  return glmm_hadd(glmm_abs(a));
}

static inline
float
glmm_norm_inf(float32x4_t a) {
  return glmm_hmax(glmm_abs(a));
}

#endif
#endif /* cglm_simd_arm_h */

#endif

#endif /* cglm_intrin_h */


#ifndef CGLM_USE_DEFAULT_EPSILON
#  ifndef GLM_FLT_EPSILON
#    define GLM_FLT_EPSILON 1e-6
#  endif
#else
#  define GLM_FLT_EPSILON FLT_EPSILON
#endif

#endif /* cglm_common_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*
 Macros:
   GLM_VEC2_ONE_INIT
   GLM_VEC2_ZERO_INIT
   GLM_VEC2_ONE
   GLM_VEC2_ZERO

 Functions:
   CGLM_INLINE void  glm_vec2(float * __restrict v, vec2 dest)
   CGLM_INLINE void  glm_vec2_copy(vec2 a, vec2 dest)
   CGLM_INLINE void  glm_vec2_zero(vec2 v)
   CGLM_INLINE void  glm_vec2_one(vec2 v)
   CGLM_INLINE float glm_vec2_dot(vec2 a, vec2 b)
   CGLM_INLINE float glm_vec2_cross(vec2 a, vec2 b)
   CGLM_INLINE float glm_vec2_norm2(vec2 v)
   CGLM_INLINE float glm_vec2_norm(vec2 vec)
   CGLM_INLINE void  glm_vec2_add(vec2 a, vec2 b, vec2 dest)
   CGLM_INLINE void  glm_vec2_adds(vec2 v, float s, vec2 dest)
   CGLM_INLINE void  glm_vec2_sub(vec2 a, vec2 b, vec2 dest)
   CGLM_INLINE void  glm_vec2_subs(vec2 v, float s, vec2 dest)
   CGLM_INLINE void  glm_vec2_mul(vec2 a, vec2 b, vec2 d)
   CGLM_INLINE void  glm_vec2_scale(vec2 v, float s, vec2 dest)
   CGLM_INLINE void  glm_vec2_scale_as(vec2 v, float s, vec2 dest)
   CGLM_INLINE void  glm_vec2_div(vec2 a, vec2 b, vec2 dest)
   CGLM_INLINE void  glm_vec2_divs(vec2 v, float s, vec2 dest)
   CGLM_INLINE void  glm_vec2_addadd(vec2 a, vec2 b, vec2 dest)
   CGLM_INLINE void  glm_vec2_subadd(vec2 a, vec2 b, vec2 dest)
   CGLM_INLINE void  glm_vec2_muladd(vec2 a, vec2 b, vec2 dest)
   CGLM_INLINE void  glm_vec2_muladds(vec2 a, float s, vec2 dest)
   CGLM_INLINE void  glm_vec2_maxadd(vec2 a, vec2 b, vec2 dest)
   CGLM_INLINE void  glm_vec2_minadd(vec2 a, vec2 b, vec2 dest)
   CGLM_INLINE void  glm_vec2_negate_to(vec2 v, vec2 dest)
   CGLM_INLINE void  glm_vec2_negate(vec2 v)
   CGLM_INLINE void  glm_vec2_normalize(vec2 v)
   CGLM_INLINE void  glm_vec2_normalize_to(vec2 vec, vec2 dest)
   CGLM_INLINE void  glm_vec2_rotate(vec2 v, float angle, vec2 dest)
   CGLM_INLINE float glm_vec2_distance2(vec2 a, vec2 b)
   CGLM_INLINE float glm_vec2_distance(vec2 a, vec2 b)
   CGLM_INLINE void  glm_vec2_maxv(vec2 v1, vec2 v2, vec2 dest)
   CGLM_INLINE void  glm_vec2_minv(vec2 v1, vec2 v2, vec2 dest)
   CGLM_INLINE void  glm_vec2_clamp(vec2 v, float minVal, float maxVal)
   CGLM_INLINE void  glm_vec2_lerp(vec2 from, vec2 to, float t, vec2 dest)

 */

#ifndef cglm_vec2_h
#define cglm_vec2_h

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*
 Functions:
   CGLM_INLINE int   glm_sign(int val);
   CGLM_INLINE float glm_signf(float val);
   CGLM_INLINE float glm_rad(float deg);
   CGLM_INLINE float glm_deg(float rad);
   CGLM_INLINE void  glm_make_rad(float *deg);
   CGLM_INLINE void  glm_make_deg(float *rad);
   CGLM_INLINE float glm_pow2(float x);
   CGLM_INLINE float glm_min(float a, float b);
   CGLM_INLINE float glm_max(float a, float b);
   CGLM_INLINE float glm_clamp(float val, float minVal, float maxVal);
   CGLM_INLINE float glm_clamp_zo(float val, float minVal, float maxVal);
   CGLM_INLINE float glm_lerp(float from, float to, float t);
   CGLM_INLINE float glm_lerpc(float from, float to, float t);
   CGLM_INLINE float glm_step(float edge, float x);
   CGLM_INLINE float glm_smooth(float t);
   CGLM_INLINE float glm_smoothstep(float edge0, float edge1, float x);
   CGLM_INLINE float glm_smoothinterp(float from, float to, float t);
   CGLM_INLINE float glm_smoothinterpc(float from, float to, float t);
   CGLM_INLINE bool  glm_eq(float a, float b);
   CGLM_INLINE float glm_percent(float from, float to, float current);
   CGLM_INLINE float glm_percentc(float from, float to, float current);
 */

#ifndef cglm_util_h
#define cglm_util_h


#define GLM_MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define GLM_MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

/*!
 * @brief get sign of 32 bit integer as +1, -1, 0
 *
 * Important: It returns 0 for zero input
 *
 * @param val integer value
 */
CGLM_INLINE
int
glm_sign(int val) {
  return ((val >> 31) - (-val >> 31));
}

/*!
 * @brief get sign of 32 bit float as +1, -1, 0
 *
 * Important: It returns 0 for zero/NaN input
 *
 * @param val float value
 */
CGLM_INLINE
float
glm_signf(float val) {
  return (float)((val > 0.0f) - (val < 0.0f));
}

/*!
 * @brief convert degree to radians
 *
 * @param[in] deg angle in degrees
 */
CGLM_INLINE
float
glm_rad(float deg) {
  return deg * GLM_PIf / 180.0f;
}

/*!
 * @brief convert radians to degree
 *
 * @param[in] rad angle in radians
 */
CGLM_INLINE
float
glm_deg(float rad) {
  return rad * 180.0f / GLM_PIf;
}

/*!
 * @brief convert exsisting degree to radians. this will override degrees value
 *
 * @param[in, out] deg pointer to angle in degrees
 */
CGLM_INLINE
void
glm_make_rad(float *deg) {
  *deg = *deg * GLM_PIf / 180.0f;
}

/*!
 * @brief convert exsisting radians to degree. this will override radians value
 *
 * @param[in, out] rad pointer to angle in radians
 */
CGLM_INLINE
void
glm_make_deg(float *rad) {
  *rad = *rad * 180.0f / GLM_PIf;
}

/*!
 * @brief multiplies given parameter with itself = x * x or powf(x, 2)
 *
 * @param[in] x x
 */
CGLM_INLINE
float
glm_pow2(float x) {
  return x * x;
}

/*!
 * @brief find minimum of given two values
 *
 * @param[in] a number 1
 * @param[in] b number 2
 */
CGLM_INLINE
float
glm_min(float a, float b) {
  if (a < b)
    return a;
  return b;
}

/*!
 * @brief find maximum of given two values
 *
 * @param[in] a number 1
 * @param[in] b number 2
 */
CGLM_INLINE
float
glm_max(float a, float b) {
  if (a > b)
    return a;
  return b;
}

/*!
 * @brief clamp a number between min and max
 *
 * @param[in] val    value to clamp
 * @param[in] minVal minimum value
 * @param[in] maxVal maximum value
 */
CGLM_INLINE
float
glm_clamp(float val, float minVal, float maxVal) {
  return glm_min(glm_max(val, minVal), maxVal);
}

/*!
 * @brief clamp a number to zero and one
 *
 * @param[in] val value to clamp
 */
CGLM_INLINE
float
glm_clamp_zo(float val) {
  return glm_clamp(val, 0.0f, 1.0f);
}

/*!
 * @brief linear interpolation between two numbers
 *
 * formula:  from + t * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount)
 */
CGLM_INLINE
float
glm_lerp(float from, float to, float t) {
  return from + t * (to - from);
}

/*!
 * @brief clamped linear interpolation between two numbers
 *
 * formula:  from + t * (to - from)
 *
 * @param[in]   from    from value
 * @param[in]   to      to value
 * @param[in]   t       interpolant (amount) clamped between 0 and 1
 */
CGLM_INLINE
float
glm_lerpc(float from, float to, float t) {
  return glm_lerp(from, to, glm_clamp_zo(t));
}

/*!
 * @brief threshold function
 *
 * @param[in]   edge    threshold
 * @param[in]   x       value to test against threshold
 * @return      returns 0.0 if x < edge, else 1.0
 */
CGLM_INLINE
float
glm_step(float edge, float x) {
  /* branching - no type conversion */
  return (x < edge) ? 0.0f : 1.0f;
  /*
   * An alternative implementation without branching
   * but with type conversion could be:
   * return !(x < edge);
   */
}

/*!
 * @brief smooth Hermite interpolation
 *
 * formula:  t^2 * (3-2t)
 *
 * @param[in]   t    interpolant (amount)
 */
CGLM_INLINE
float
glm_smooth(float t) {
  return t * t * (3.0f - 2.0f * t);
}

/*!
 * @brief threshold function with a smooth transition (according to OpenCL specs)
 *
 * formula:  t^2 * (3-2t)
 *
 * @param[in]   edge0 low threshold
 * @param[in]   edge1 high threshold
 * @param[in]   x     interpolant (amount)
 */
CGLM_INLINE
float
glm_smoothstep(float edge0, float edge1, float x) {
  float t;
  t = glm_clamp_zo((x - edge0) / (edge1 - edge0));
  return glm_smooth(t);
}

/*!
 * @brief smoothstep interpolation between two numbers
 *
 * formula:  from + smoothstep(t) * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount)
 */
CGLM_INLINE
float
glm_smoothinterp(float from, float to, float t) {
  return from + glm_smooth(t) * (to - from);
}

/*!
 * @brief clamped smoothstep interpolation between two numbers
 *
 * formula:  from + smoothstep(t) * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount) clamped between 0 and 1
 */
CGLM_INLINE
float
glm_smoothinterpc(float from, float to, float t) {
  return glm_smoothinterp(from, to, glm_clamp_zo(t));
}

/*!
 * @brief check if two float equal with using EPSILON
 *
 * @param[in]   a   a
 * @param[in]   b   b
 */
CGLM_INLINE
bool
glm_eq(float a, float b) {
  return fabsf(a - b) <= GLM_FLT_EPSILON;
}

/*!
 * @brief percentage of current value between start and end value
 *
 * maybe fraction could be alternative name.
 *
 * @param[in]   from    from value
 * @param[in]   to      to value
 * @param[in]   current current value
 */
CGLM_INLINE
float
glm_percent(float from, float to, float current) {
  float t;

  if ((t = to - from) == 0.0f)
    return 1.0f;

  return (current - from) / t;
}

/*!
 * @brief clamped percentage of current value between start and end value
 *
 * @param[in]   from    from value
 * @param[in]   to      to value
 * @param[in]   current current value
 */
CGLM_INLINE
float
glm_percentc(float from, float to, float current) {
  return glm_clamp_zo(glm_percent(from, to, current));
}

/*!
* @brief swap two float values
*
* @param[in]   a float value 1 (pointer)
* @param[in]   b float value 2 (pointer)
*/
CGLM_INLINE
void
glm_swapf(float * __restrict a, float * __restrict b) {
  float t;
  t  = *a;
  *a = *b;
  *b = t;
}

#endif /* cglm_util_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*
 Functions:
   CGLM_INLINE void  glm_vec2_fill(vec2 v, float val)
   CGLM_INLINE bool  glm_vec2_eq(vec2 v, float val);
   CGLM_INLINE bool  glm_vec2_eq_eps(vec2 v, float val);
   CGLM_INLINE bool  glm_vec2_eq_all(vec2 v);
   CGLM_INLINE bool  glm_vec2_eqv(vec2 a, vec2 b);
   CGLM_INLINE bool  glm_vec2_eqv_eps(vec2 a, vec2 b);
   CGLM_INLINE float glm_vec2_max(vec2 v);
   CGLM_INLINE float glm_vec2_min(vec2 v);
   CGLM_INLINE bool  glm_vec2_isnan(vec2 v);
   CGLM_INLINE bool  glm_vec2_isinf(vec2 v);
   CGLM_INLINE bool  glm_vec2_isvalid(vec2 v);
   CGLM_INLINE void  glm_vec2_sign(vec2 v, vec2 dest);
   CGLM_INLINE void  glm_vec2_sqrt(vec2 v, vec2 dest);
 */

#ifndef cglm_vec2_ext_h
#define cglm_vec2_ext_h


/*!
 * @brief fill a vector with specified value
 *
 * @param[out] v   dest
 * @param[in]  val value
 */
CGLM_INLINE
void
glm_vec2_fill(vec2 v, float val) {
  v[0] = v[1] = val;
}

/*!
 * @brief check if vector is equal to value (without epsilon)
 *
 * @param[in] v   vector
 * @param[in] val value
 */
CGLM_INLINE
bool
glm_vec2_eq(vec2 v, float val) {
  return v[0] == val && v[0] == v[1];
}

/*!
 * @brief check if vector is equal to value (with epsilon)
 *
 * @param[in] v   vector
 * @param[in] val value
 */
CGLM_INLINE
bool
glm_vec2_eq_eps(vec2 v, float val) {
  return fabsf(v[0] - val) <= GLM_FLT_EPSILON
         && fabsf(v[1] - val) <= GLM_FLT_EPSILON;
}

/*!
 * @brief check if vectors members are equal (without epsilon)
 *
 * @param[in] v   vector
 */
CGLM_INLINE
bool
glm_vec2_eq_all(vec2 v) {
  return glm_vec2_eq_eps(v, v[0]);
}

/*!
 * @brief check if vector is equal to another (without epsilon)
 *
 * @param[in] a vector
 * @param[in] b vector
 */
CGLM_INLINE
bool
glm_vec2_eqv(vec2 a, vec2 b) {
  return a[0] == b[0] && a[1] == b[1];
}

/*!
 * @brief check if vector is equal to another (with epsilon)
 *
 * @param[in] a vector
 * @param[in] b vector
 */
CGLM_INLINE
bool
glm_vec2_eqv_eps(vec2 a, vec2 b) {
  return fabsf(a[0] - b[0]) <= GLM_FLT_EPSILON
         && fabsf(a[1] - b[1]) <= GLM_FLT_EPSILON;
}

/*!
 * @brief max value of vector
 *
 * @param[in] v vector
 */
CGLM_INLINE
float
glm_vec2_max(vec2 v) {
  return glm_max(v[0], v[1]);
}

/*!
 * @brief min value of vector
 *
 * @param[in] v vector
 */
CGLM_INLINE
float
glm_vec2_min(vec2 v) {
  return glm_min(v[0], v[1]);
}

/*!
 * @brief check if all items are NaN (not a number)
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
CGLM_INLINE
bool
glm_vec2_isnan(vec2 v) {
  return isnan(v[0]) || isnan(v[1]);
}

/*!
 * @brief check if all items are INFINITY
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
CGLM_INLINE
bool
glm_vec2_isinf(vec2 v) {
  return isinf(v[0]) || isinf(v[1]);
}

/*!
 * @brief check if all items are valid number
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
CGLM_INLINE
bool
glm_vec2_isvalid(vec2 v) {
  return !glm_vec2_isnan(v) && !glm_vec2_isinf(v);
}

/*!
 * @brief get sign of 32 bit float as +1, -1, 0
 *
 * Important: It returns 0 for zero/NaN input
 *
 * @param v vector
 */
CGLM_INLINE
void
glm_vec2_sign(vec2 v, vec2 dest) {
  dest[0] = glm_signf(v[0]);
  dest[1] = glm_signf(v[1]);
}

/*!
 * @brief square root of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec2_sqrt(vec2 v, vec2 dest) {
  dest[0] = sqrtf(v[0]);
  dest[1] = sqrtf(v[1]);
}

#endif /* cglm_vec2_ext_h */


#define GLM_VEC2_ONE_INIT   {1.0f, 1.0f}
#define GLM_VEC2_ZERO_INIT  {0.0f, 0.0f}

#define GLM_VEC2_ONE  ((vec2)GLM_VEC2_ONE_INIT)
#define GLM_VEC2_ZERO ((vec2)GLM_VEC2_ZERO_INIT)

/*!
 * @brief init vec2 using another vector
 *
 * @param[in]  v    a vector
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec2(float * __restrict v, vec2 dest) {
  dest[0] = v[0];
  dest[1] = v[1];
}

/*!
 * @brief copy all members of [a] to [dest]
 *
 * @param[in]  a    source
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec2_copy(vec2 a, vec2 dest) {
  dest[0] = a[0];
  dest[1] = a[1];
}

/*!
 * @brief make vector zero
 *
 * @param[in, out]  v vector
 */
CGLM_INLINE
void
glm_vec2_zero(vec2 v) {
  v[0] = v[1] = 0.0f;
}

/*!
 * @brief make vector one
 *
 * @param[in, out]  v vector
 */
CGLM_INLINE
void
glm_vec2_one(vec2 v) {
  v[0] = v[1] = 1.0f;
}

/*!
 * @brief vec2 dot product
 *
 * @param[in] a vector1
 * @param[in] b vector2
 *
 * @return dot product
 */
CGLM_INLINE
float
glm_vec2_dot(vec2 a, vec2 b) {
  return a[0] * b[0] + a[1] * b[1];
}

/*!
 * @brief vec2 cross product
 *
 * REF: http://allenchou.net/2013/07/cross-product-of-2d-vectors/
 *
 * @param[in]  a vector1
 * @param[in]  b vector2
 *
 * @return Z component of cross product
 */
CGLM_INLINE
float
glm_vec2_cross(vec2 a, vec2 b) {
  /* just calculate the z-component */
  return a[0] * b[1] - a[1] * b[0];
}

/*!
 * @brief norm * norm (magnitude) of vec
 *
 * we can use this func instead of calling norm * norm, because it would call
 * sqrtf fuction twice but with this func we can avoid func call, maybe this is
 * not good name for this func
 *
 * @param[in] v vector
 *
 * @return norm * norm
 */
CGLM_INLINE
float
glm_vec2_norm2(vec2 v) {
  return glm_vec2_dot(v, v);
}

/*!
 * @brief norm (magnitude) of vec2
 *
 * @param[in] vec vector
 *
 * @return norm
 */
CGLM_INLINE
float
glm_vec2_norm(vec2 vec) {
  return sqrtf(glm_vec2_norm2(vec));
}

/*!
 * @brief add a vector to b vector store result in dest
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec2_add(vec2 a, vec2 b, vec2 dest) {
  dest[0] = a[0] + b[0];
  dest[1] = a[1] + b[1];
}

/*!
 * @brief add scalar to v vector store result in dest (d = v + s)
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec2_adds(vec2 v, float s, vec2 dest) {
  dest[0] = v[0] + s;
  dest[1] = v[1] + s;
}

/*!
 * @brief subtract b vector from a vector store result in dest
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec2_sub(vec2 a, vec2 b, vec2 dest) {
  dest[0] = a[0] - b[0];
  dest[1] = a[1] - b[1];
}

/*!
 * @brief subtract scalar from v vector store result in dest (d = v - s)
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec2_subs(vec2 v, float s, vec2 dest) {
  dest[0] = v[0] - s;
  dest[1] = v[1] - s;
}

/*!
 * @brief multiply two vector (component-wise multiplication)
 *
 * @param a    v1
 * @param b    v2
 * @param dest v3 = (a[0] * b[0], a[1] * b[1])
 */
CGLM_INLINE
void
glm_vec2_mul(vec2 a, vec2 b, vec2 dest) {
  dest[0] = a[0] * b[0];
  dest[1] = a[1] * b[1];
}

/*!
 * @brief multiply/scale vector with scalar: result = v * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec2_scale(vec2 v, float s, vec2 dest) {
  dest[0] = v[0] * s;
  dest[1] = v[1] * s;
}

/*!
 * @brief scale as vector specified: result = unit(v) * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec2_scale_as(vec2 v, float s, vec2 dest) {
  float norm;
  norm = glm_vec2_norm(v);

  if (norm == 0.0f) {
    glm_vec2_zero(dest);
    return;
  }

  glm_vec2_scale(v, s / norm, dest);
}

/*!
 * @brief div vector with another component-wise division: d = a / b
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest result = (a[0]/b[0], a[1]/b[1])
 */
CGLM_INLINE
void
glm_vec2_div(vec2 a, vec2 b, vec2 dest) {
  dest[0] = a[0] / b[0];
  dest[1] = a[1] / b[1];
}

/*!
 * @brief div vector with scalar: d = v / s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest result = (a[0]/s, a[1]/s)
 */
CGLM_INLINE
void
glm_vec2_divs(vec2 v, float s, vec2 dest) {
  dest[0] = v[0] / s;
  dest[1] = v[1] / s;
}

/*!
 * @brief add two vectors and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a + b)
 */
CGLM_INLINE
void
glm_vec2_addadd(vec2 a, vec2 b, vec2 dest) {
  dest[0] += a[0] + b[0];
  dest[1] += a[1] + b[1];
}

/*!
 * @brief sub two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a + b)
 */
CGLM_INLINE
void
glm_vec2_subadd(vec2 a, vec2 b, vec2 dest) {
  dest[0] += a[0] - b[0];
  dest[1] += a[1] - b[1];
}

/*!
 * @brief mul two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a * b)
 */
CGLM_INLINE
void
glm_vec2_muladd(vec2 a, vec2 b, vec2 dest) {
  dest[0] += a[0] * b[0];
  dest[1] += a[1] * b[1];
}

/*!
 * @brief mul vector with scalar and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest += (a * b)
 */
CGLM_INLINE
void
glm_vec2_muladds(vec2 a, float s, vec2 dest) {
  dest[0] += a[0] * s;
  dest[1] += a[1] * s;
}

/*!
 * @brief add max of two vector to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += max(a, b)
 */
CGLM_INLINE
void
glm_vec2_maxadd(vec2 a, vec2 b, vec2 dest) {
  dest[0] += glm_max(a[0], b[0]);
  dest[1] += glm_max(a[1], b[1]);
}

/*!
 * @brief add min of two vector to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += min(a, b)
 */
CGLM_INLINE
void
glm_vec2_minadd(vec2 a, vec2 b, vec2 dest) {
  dest[0] += glm_min(a[0], b[0]);
  dest[1] += glm_min(a[1], b[1]);
}

/*!
 * @brief negate vector components and store result in dest
 *
 * @param[in]   v     vector
 * @param[out]  dest  result vector
 */
CGLM_INLINE
void
glm_vec2_negate_to(vec2 v, vec2 dest) {
  dest[0] = -v[0];
  dest[1] = -v[1];
}

/*!
 * @brief negate vector components
 *
 * @param[in, out]  v  vector
 */
CGLM_INLINE
void
glm_vec2_negate(vec2 v) {
  glm_vec2_negate_to(v, v);
}

/*!
 * @brief normalize vector and store result in same vec
 *
 * @param[in, out] v vector
 */
CGLM_INLINE
void
glm_vec2_normalize(vec2 v) {
  float norm;

  norm = glm_vec2_norm(v);

  if (norm == 0.0f) {
    v[0] = v[1] = 0.0f;
    return;
  }

  glm_vec2_scale(v, 1.0f / norm, v);
}

/*!
 * @brief normalize vector to dest
 *
 * @param[in]  v    source
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec2_normalize_to(vec2 v, vec2 dest) {
  float norm;

  norm = glm_vec2_norm(v);

  if (norm == 0.0f) {
    glm_vec2_zero(dest);
    return;
  }

  glm_vec2_scale(v, 1.0f / norm, dest);
}

/*!
 * @brief rotate vec2 around origin by angle (CCW: counterclockwise)
 *
 * Formula:
 *   𝑥2 = cos(a)𝑥1 − sin(a)𝑦1
 *   𝑦2 = sin(a)𝑥1 + cos(a)𝑦1
 *
 * @param[in]  v     vector to rotate
 * @param[in]  angle angle by radians
 * @param[out] dest  destination vector
 */
CGLM_INLINE
void
glm_vec2_rotate(vec2 v, float angle, vec2 dest) {
  float c, s, x1, y1;

  c  = cosf(angle);
  s  = sinf(angle);

  x1 = v[0];
  y1 = v[1];

  dest[0] = c * x1 - s * y1;
  dest[1] = s * x1 + c * y1;
}

/**
 * @brief squared distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return returns squared distance (distance * distance)
 */
CGLM_INLINE
float
glm_vec2_distance2(vec2 a, vec2 b) {
  return glm_pow2(b[0] - a[0]) + glm_pow2(b[1] - a[1]);
}

/**
 * @brief distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return returns distance
 */
CGLM_INLINE
float
glm_vec2_distance(vec2 a, vec2 b) {
  return sqrtf(glm_vec2_distance2(a, b));
}

/*!
 * @brief max values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec2_maxv(vec2 a, vec2 b, vec2 dest) {
  dest[0] = glm_max(a[0], b[0]);
  dest[1] = glm_max(a[1], b[1]);
}

/*!
 * @brief min values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec2_minv(vec2 a, vec2 b, vec2 dest) {
  dest[0] = glm_min(a[0], b[0]);
  dest[1] = glm_min(a[1], b[1]);
}

/*!
 * @brief clamp vector's individual members between min and max values
 *
 * @param[in, out]  v      vector
 * @param[in]       minval minimum value
 * @param[in]       maxval maximum value
 */
CGLM_INLINE
void
glm_vec2_clamp(vec2 v, float minval, float maxval) {
  v[0] = glm_clamp(v[0], minval, maxval);
  v[1] = glm_clamp(v[1], minval, maxval);
}

/*!
 * @brief linear interpolation between two vector
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount) clamped between 0 and 1
 * @param[out]  dest destination
 */
CGLM_INLINE
void
glm_vec2_lerp(vec2 from, vec2 to, float t, vec2 dest) {
  vec2 s, v;

  /* from + s * (to - from) */
  glm_vec2_fill(s, glm_clamp_zo(t));
  glm_vec2_sub(to, from, v);
  glm_vec2_mul(s, v, v);
  glm_vec2_add(from, v, dest);
}

#endif /* cglm_vec2_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*
 Macros:
   GLM_VEC3_ONE_INIT
   GLM_VEC3_ZERO_INIT
   GLM_VEC3_ONE
   GLM_VEC3_ZERO
   GLM_YUP
   GLM_ZUP
   GLM_XUP

 Functions:
   CGLM_INLINE void  glm_vec3(vec4 v4, vec3 dest);
   CGLM_INLINE void  glm_vec3_copy(vec3 a, vec3 dest);
   CGLM_INLINE void  glm_vec3_zero(vec3 v);
   CGLM_INLINE void  glm_vec3_one(vec3 v);
   CGLM_INLINE float glm_vec3_dot(vec3 a, vec3 b);
   CGLM_INLINE float glm_vec3_norm2(vec3 v);
   CGLM_INLINE float glm_vec3_norm(vec3 v);
   CGLM_INLINE float glm_vec3_norm_one(vec3 v);
   CGLM_INLINE float glm_vec3_norm_inf(vec3 v);
   CGLM_INLINE void  glm_vec3_add(vec3 a, vec3 b, vec3 dest);
   CGLM_INLINE void  glm_vec3_adds(vec3 a, float s, vec3 dest);
   CGLM_INLINE void  glm_vec3_sub(vec3 a, vec3 b, vec3 dest);
   CGLM_INLINE void  glm_vec3_subs(vec3 a, float s, vec3 dest);
   CGLM_INLINE void  glm_vec3_mul(vec3 a, vec3 b, vec3 dest);
   CGLM_INLINE void  glm_vec3_scale(vec3 v, float s, vec3 dest);
   CGLM_INLINE void  glm_vec3_scale_as(vec3 v, float s, vec3 dest);
   CGLM_INLINE void  glm_vec3_div(vec3 a, vec3 b, vec3 dest);
   CGLM_INLINE void  glm_vec3_divs(vec3 a, float s, vec3 dest);
   CGLM_INLINE void  glm_vec3_addadd(vec3 a, vec3 b, vec3 dest);
   CGLM_INLINE void  glm_vec3_subadd(vec3 a, vec3 b, vec3 dest);
   CGLM_INLINE void  glm_vec3_muladd(vec3 a, vec3 b, vec3 dest);
   CGLM_INLINE void  glm_vec3_muladds(vec3 a, float s, vec3 dest);
   CGLM_INLINE void  glm_vec3_maxadd(vec3 a, vec3 b, vec3 dest);
   CGLM_INLINE void  glm_vec3_minadd(vec3 a, vec3 b, vec3 dest);
   CGLM_INLINE void  glm_vec3_flipsign(vec3 v);
   CGLM_INLINE void  glm_vec3_flipsign_to(vec3 v, vec3 dest);
   CGLM_INLINE void  glm_vec3_negate_to(vec3 v, vec3 dest);
   CGLM_INLINE void  glm_vec3_negate(vec3 v);
   CGLM_INLINE void  glm_vec3_inv(vec3 v);
   CGLM_INLINE void  glm_vec3_inv_to(vec3 v, vec3 dest);
   CGLM_INLINE void  glm_vec3_normalize(vec3 v);
   CGLM_INLINE void  glm_vec3_normalize_to(vec3 v, vec3 dest);
   CGLM_INLINE void  glm_vec3_cross(vec3 a, vec3 b, vec3 d);
   CGLM_INLINE void  glm_vec3_crossn(vec3 a, vec3 b, vec3 dest);
   CGLM_INLINE float glm_vec3_angle(vec3 a, vec3 b);
   CGLM_INLINE void  glm_vec3_rotate(vec3 v, float angle, vec3 axis);
   CGLM_INLINE void  glm_vec3_rotate_m4(mat4 m, vec3 v, vec3 dest);
   CGLM_INLINE void  glm_vec3_rotate_m3(mat3 m, vec3 v, vec3 dest);
   CGLM_INLINE void  glm_vec3_proj(vec3 a, vec3 b, vec3 dest);
   CGLM_INLINE void  glm_vec3_center(vec3 a, vec3 b, vec3 dest);
   CGLM_INLINE float glm_vec3_distance(vec3 a, vec3 b);
   CGLM_INLINE float glm_vec3_distance2(vec3 a, vec3 b);
   CGLM_INLINE void  glm_vec3_maxv(vec3 a, vec3 b, vec3 dest);
   CGLM_INLINE void  glm_vec3_minv(vec3 a, vec3 b, vec3 dest);
   CGLM_INLINE void  glm_vec3_ortho(vec3 v, vec3 dest);
   CGLM_INLINE void  glm_vec3_clamp(vec3 v, float minVal, float maxVal);
   CGLM_INLINE void  glm_vec3_lerp(vec3 from, vec3 to, float t, vec3 dest);
   CGLM_INLINE void  glm_vec3_lerpc(vec3 from, vec3 to, float t, vec3 dest);
   CGLM_INLINE void  glm_vec3_mix(vec3 from, vec3 to, float t, vec3 dest);
   CGLM_INLINE void  glm_vec3_mixc(vec3 from, vec3 to, float t, vec3 dest);
   CGLM_INLINE void  glm_vec3_step_uni(float edge, vec3 x, vec3 dest);
   CGLM_INLINE void  glm_vec3_step(vec3 edge, vec3 x, vec3 dest);
   CGLM_INLINE void  glm_vec3_smoothstep_uni(float edge0, float edge1, vec3 x, vec3 dest);
   CGLM_INLINE void  glm_vec3_smoothstep(vec3 edge0, vec3 edge1, vec3 x, vec3 dest);
   CGLM_INLINE void  glm_vec3_smoothinterp(vec3 from, vec3 to, float t, vec3 dest);
   CGLM_INLINE void  glm_vec3_smoothinterpc(vec3 from, vec3 to, float t, vec3 dest);
   CGLM_INLINE void  glm_vec3_swizzle(vec3 v, int mask, vec3 dest);

 Convenient:
   CGLM_INLINE void  glm_cross(vec3 a, vec3 b, vec3 d);
   CGLM_INLINE float glm_dot(vec3 a, vec3 b);
   CGLM_INLINE void  glm_normalize(vec3 v);
   CGLM_INLINE void  glm_normalize_to(vec3 v, vec3 dest);

 DEPRECATED:
   glm_vec3_dup
   glm_vec3_flipsign
   glm_vec3_flipsign_to
   glm_vec3_inv
   glm_vec3_inv_to
   glm_vec3_mulv
 */

#ifndef cglm_vec3_h
#define cglm_vec3_h

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*
 Macros:
   GLM_VEC4_ONE_INIT
   GLM_VEC4_BLACK_INIT
   GLM_VEC4_ZERO_INIT
   GLM_VEC4_ONE
   GLM_VEC4_BLACK
   GLM_VEC4_ZERO

 Functions:
   CGLM_INLINE void  glm_vec4(vec3 v3, float last, vec4 dest);
   CGLM_INLINE void  glm_vec4_copy3(vec4 a, vec3 dest);
   CGLM_INLINE void  glm_vec4_copy(vec4 v, vec4 dest);
   CGLM_INLINE void  glm_vec4_ucopy(vec4 v, vec4 dest);
   CGLM_INLINE float glm_vec4_dot(vec4 a, vec4 b);
   CGLM_INLINE float glm_vec4_norm2(vec4 v);
   CGLM_INLINE float glm_vec4_norm(vec4 v);
   CGLM_INLINE float glm_vec4_norm_one(vec4 v);
   CGLM_INLINE float glm_vec4_norm_inf(vec4 v);
   CGLM_INLINE void  glm_vec4_add(vec4 a, vec4 b, vec4 dest);
   CGLM_INLINE void  glm_vec4_adds(vec4 v, float s, vec4 dest);
   CGLM_INLINE void  glm_vec4_sub(vec4 a, vec4 b, vec4 dest);
   CGLM_INLINE void  glm_vec4_subs(vec4 v, float s, vec4 dest);
   CGLM_INLINE void  glm_vec4_mul(vec4 a, vec4 b, vec4 dest);
   CGLM_INLINE void  glm_vec4_scale(vec4 v, float s, vec4 dest);
   CGLM_INLINE void  glm_vec4_scale_as(vec4 v, float s, vec4 dest);
   CGLM_INLINE void  glm_vec4_div(vec4 a, vec4 b, vec4 dest);
   CGLM_INLINE void  glm_vec4_divs(vec4 v, float s, vec4 dest);
   CGLM_INLINE void  glm_vec4_addadd(vec4 a, vec4 b, vec4 dest);
   CGLM_INLINE void  glm_vec4_subadd(vec4 a, vec4 b, vec4 dest);
   CGLM_INLINE void  glm_vec4_muladd(vec4 a, vec4 b, vec4 dest);
   CGLM_INLINE void  glm_vec4_muladds(vec4 a, float s, vec4 dest);
   CGLM_INLINE void  glm_vec4_maxadd(vec4 a, vec4 b, vec4 dest);
   CGLM_INLINE void  glm_vec4_minadd(vec4 a, vec4 b, vec4 dest);
   CGLM_INLINE void  glm_vec4_negate(vec4 v);
   CGLM_INLINE void  glm_vec4_inv(vec4 v);
   CGLM_INLINE void  glm_vec4_inv_to(vec4 v, vec4 dest);
   CGLM_INLINE void  glm_vec4_normalize(vec4 v);
   CGLM_INLINE void  glm_vec4_normalize_to(vec4 vec, vec4 dest);
   CGLM_INLINE float glm_vec4_distance(vec4 a, vec4 b);
   CGLM_INLINE float glm_vec4_distance2(vec4 a, vec4 b);
   CGLM_INLINE void  glm_vec4_maxv(vec4 a, vec4 b, vec4 dest);
   CGLM_INLINE void  glm_vec4_minv(vec4 a, vec4 b, vec4 dest);
   CGLM_INLINE void  glm_vec4_clamp(vec4 v, float minVal, float maxVal);
   CGLM_INLINE void  glm_vec4_lerp(vec4 from, vec4 to, float t, vec4 dest);
   CGLM_INLINE void  glm_vec4_lerpc(vec4 from, vec4 to, float t, vec4 dest);
   CGLM_INLINE void  glm_vec4_step_uni(float edge, vec4 x, vec4 dest);
   CGLM_INLINE void  glm_vec4_step(vec4 edge, vec4 x, vec4 dest);
   CGLM_INLINE void  glm_vec4_smoothstep_uni(float edge0, float edge1, vec4 x, vec4 dest);
   CGLM_INLINE void  glm_vec4_smoothstep(vec4 edge0, vec4 edge1, vec4 x, vec4 dest);
   CGLM_INLINE void  glm_vec4_smoothinterp(vec4 from, vec4 to, float t, vec4 dest);
   CGLM_INLINE void  glm_vec4_smoothinterpc(vec4 from, vec4 to, float t, vec4 dest);
   CGLM_INLINE void  glm_vec4_swizzle(vec4 v, int mask, vec4 dest);

 DEPRECATED:
   glm_vec4_dup
   glm_vec4_flipsign
   glm_vec4_flipsign_to
   glm_vec4_inv
   glm_vec4_inv_to
   glm_vec4_mulv
 */

#ifndef cglm_vec4_h
#define cglm_vec4_h

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*!
 * @brief SIMD like functions
 */

/*
 Functions:
   CGLM_INLINE void  glm_vec4_broadcast(float val, vec4 d);
   CGLM_INLINE void  glm_vec4_fill(vec4 v, float val);
   CGLM_INLINE bool  glm_vec4_eq(vec4 v, float val);
   CGLM_INLINE bool  glm_vec4_eq_eps(vec4 v, float val);
   CGLM_INLINE bool  glm_vec4_eq_all(vec4 v);
   CGLM_INLINE bool  glm_vec4_eqv(vec4 a, vec4 b);
   CGLM_INLINE bool  glm_vec4_eqv_eps(vec4 a, vec4 b);
   CGLM_INLINE float glm_vec4_max(vec4 v);
   CGLM_INLINE float glm_vec4_min(vec4 v);
   CGLM_INLINE bool  glm_vec4_isnan(vec4 v);
   CGLM_INLINE bool  glm_vec4_isinf(vec4 v);
   CGLM_INLINE bool  glm_vec4_isvalid(vec4 v);
   CGLM_INLINE void  glm_vec4_sign(vec4 v, vec4 dest);
   CGLM_INLINE void  glm_vec4_abs(vec4 v, vec4 dest);
   CGLM_INLINE void  glm_vec4_fract(vec4 v, vec4 dest);
   CGLM_INLINE float glm_vec4_hadd(vec4 v);
   CGLM_INLINE void  glm_vec4_sqrt(vec4 v, vec4 dest);
 */

#ifndef cglm_vec4_ext_h
#define cglm_vec4_ext_h

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*!
 * @brief SIMD like functions
 */

/*
 Functions:
   CGLM_INLINE void  glm_vec3_broadcast(float val, vec3 d);
   CGLM_INLINE void  glm_vec3_fill(vec3 v, float val);
   CGLM_INLINE bool  glm_vec3_eq(vec3 v, float val);
   CGLM_INLINE bool  glm_vec3_eq_eps(vec3 v, float val);
   CGLM_INLINE bool  glm_vec3_eq_all(vec3 v);
   CGLM_INLINE bool  glm_vec3_eqv(vec3 a, vec3 b);
   CGLM_INLINE bool  glm_vec3_eqv_eps(vec3 a, vec3 b);
   CGLM_INLINE float glm_vec3_max(vec3 v);
   CGLM_INLINE float glm_vec3_min(vec3 v);
   CGLM_INLINE bool  glm_vec3_isnan(vec3 v);
   CGLM_INLINE bool  glm_vec3_isinf(vec3 v);
   CGLM_INLINE bool  glm_vec3_isvalid(vec3 v);
   CGLM_INLINE void  glm_vec3_sign(vec3 v, vec3 dest);
   CGLM_INLINE void  glm_vec3_abs(vec3 v, vec3 dest);
   CGLM_INLINE void  glm_vec3_fract(vec3 v, vec3 dest);
   CGLM_INLINE float glm_vec3_hadd(vec3 v);
   CGLM_INLINE void  glm_vec3_sqrt(vec3 v, vec3 dest);
 */

#ifndef cglm_vec3_ext_h
#define cglm_vec3_ext_h


/*!
 * @brief fill a vector with specified value
 *
 * @param[in]  val value
 * @param[out] d   dest
 */
CGLM_INLINE
void
glm_vec3_broadcast(float val, vec3 d) {
  d[0] = d[1] = d[2] = val;
}

/*!
 * @brief fill a vector with specified value
 *
 * @param[out] v   dest
 * @param[in]  val value
 */
CGLM_INLINE
void
glm_vec3_fill(vec3 v, float val) {
  v[0] = v[1] = v[2] = val;
}

/*!
 * @brief check if vector is equal to value (without epsilon)
 *
 * @param[in] v   vector
 * @param[in] val value
 */
CGLM_INLINE
bool
glm_vec3_eq(vec3 v, float val) {
  return v[0] == val && v[0] == v[1] && v[0] == v[2];
}

/*!
 * @brief check if vector is equal to value (with epsilon)
 *
 * @param[in] v   vector
 * @param[in] val value
 */
CGLM_INLINE
bool
glm_vec3_eq_eps(vec3 v, float val) {
  return fabsf(v[0] - val) <= GLM_FLT_EPSILON
         && fabsf(v[1] - val) <= GLM_FLT_EPSILON
         && fabsf(v[2] - val) <= GLM_FLT_EPSILON;
}

/*!
 * @brief check if vectors members are equal (without epsilon)
 *
 * @param[in] v   vector
 */
CGLM_INLINE
bool
glm_vec3_eq_all(vec3 v) {
  return glm_vec3_eq_eps(v, v[0]);
}

/*!
 * @brief check if vector is equal to another (without epsilon)
 *
 * @param[in] a vector
 * @param[in] b vector
 */
CGLM_INLINE
bool
glm_vec3_eqv(vec3 a, vec3 b) {
  return a[0] == b[0]
         && a[1] == b[1]
         && a[2] == b[2];
}

/*!
 * @brief check if vector is equal to another (with epsilon)
 *
 * @param[in] a vector
 * @param[in] b vector
 */
CGLM_INLINE
bool
glm_vec3_eqv_eps(vec3 a, vec3 b) {
  return fabsf(a[0] - b[0]) <= GLM_FLT_EPSILON
         && fabsf(a[1] - b[1]) <= GLM_FLT_EPSILON
         && fabsf(a[2] - b[2]) <= GLM_FLT_EPSILON;
}

/*!
 * @brief max value of vector
 *
 * @param[in] v vector
 */
CGLM_INLINE
float
glm_vec3_max(vec3 v) {
  float max;

  max = v[0];
  if (v[1] > max)
    max = v[1];
  if (v[2] > max)
    max = v[2];

  return max;
}

/*!
 * @brief min value of vector
 *
 * @param[in] v vector
 */
CGLM_INLINE
float
glm_vec3_min(vec3 v) {
  float min;

  min = v[0];
  if (v[1] < min)
    min = v[1];
  if (v[2] < min)
    min = v[2];

  return min;
}

/*!
 * @brief check if all items are NaN (not a number)
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
CGLM_INLINE
bool
glm_vec3_isnan(vec3 v) {
  return isnan(v[0]) || isnan(v[1]) || isnan(v[2]);
}

/*!
 * @brief check if all items are INFINITY
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
CGLM_INLINE
bool
glm_vec3_isinf(vec3 v) {
  return isinf(v[0]) || isinf(v[1]) || isinf(v[2]);
}

/*!
 * @brief check if all items are valid number
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
CGLM_INLINE
bool
glm_vec3_isvalid(vec3 v) {
  return !glm_vec3_isnan(v) && !glm_vec3_isinf(v);
}

/*!
 * @brief get sign of 32 bit float as +1, -1, 0
 *
 * Important: It returns 0 for zero/NaN input
 *
 * @param v vector
 */
CGLM_INLINE
void
glm_vec3_sign(vec3 v, vec3 dest) {
  dest[0] = glm_signf(v[0]);
  dest[1] = glm_signf(v[1]);
  dest[2] = glm_signf(v[2]);
}

/*!
 * @brief absolute value of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec3_abs(vec3 v, vec3 dest) {
  dest[0] = fabsf(v[0]);
  dest[1] = fabsf(v[1]);
  dest[2] = fabsf(v[2]);
}

/*!
 * @brief fractional part of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec3_fract(vec3 v, vec3 dest) {
  dest[0] = fminf(v[0] - floorf(v[0]), 0.999999940395355224609375f);
  dest[1] = fminf(v[1] - floorf(v[1]), 0.999999940395355224609375f);
  dest[2] = fminf(v[2] - floorf(v[2]), 0.999999940395355224609375f);
}

/*!
 * @brief vector reduction by summation
 * @warning could overflow
 *
 * @param[in]  v    vector
 * @return     sum of all vector's elements
 */
CGLM_INLINE
float
glm_vec3_hadd(vec3 v) {
  return v[0] + v[1] + v[2];
}

/*!
 * @brief square root of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec3_sqrt(vec3 v, vec3 dest) {
  dest[0] = sqrtf(v[0]);
  dest[1] = sqrtf(v[1]);
  dest[2] = sqrtf(v[2]);
}

#endif /* cglm_vec3_ext_h */


/*!
 * @brief fill a vector with specified value
 *
 * @param val value
 * @param d   dest
 */
CGLM_INLINE
void
glm_vec4_broadcast(float val, vec4 d) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(d, _mm_set1_ps(val));
#else
  d[0] = d[1] = d[2] = d[3] = val;
#endif
}

/*!
 * @brief fill a vector with specified value
 *
 * @param v   dest
 * @param val value
 */
CGLM_INLINE
void
glm_vec4_fill(vec4 v, float val) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(v, _mm_set1_ps(val));
#else
  v[0] = v[1] = v[2] = v[3] = val;
#endif
}

/*!
 * @brief check if vector is equal to value (without epsilon)
 *
 * @param v   vector
 * @param val value
 */
CGLM_INLINE
bool
glm_vec4_eq(vec4 v, float val) {
  return v[0] == val
         && v[0] == v[1]
         && v[0] == v[2]
         && v[0] == v[3];
}

/*!
 * @brief check if vector is equal to value (with epsilon)
 *
 * @param v   vector
 * @param val value
 */
CGLM_INLINE
bool
glm_vec4_eq_eps(vec4 v, float val) {
  return fabsf(v[0] - val) <= GLM_FLT_EPSILON
         && fabsf(v[1] - val) <= GLM_FLT_EPSILON
         && fabsf(v[2] - val) <= GLM_FLT_EPSILON
         && fabsf(v[3] - val) <= GLM_FLT_EPSILON;
}

/*!
 * @brief check if vectors members are equal (without epsilon)
 *
 * @param v   vector
 */
CGLM_INLINE
bool
glm_vec4_eq_all(vec4 v) {
  return glm_vec4_eq_eps(v, v[0]);
}

/*!
 * @brief check if vector is equal to another (without epsilon)
 *
 * @param a vector
 * @param b vector
 */
CGLM_INLINE
bool
glm_vec4_eqv(vec4 a, vec4 b) {
  return a[0] == b[0]
         && a[1] == b[1]
         && a[2] == b[2]
         && a[3] == b[3];
}

/*!
 * @brief check if vector is equal to another (with epsilon)
 *
 * @param a vector
 * @param b vector
 */
CGLM_INLINE
bool
glm_vec4_eqv_eps(vec4 a, vec4 b) {
  return fabsf(a[0] - b[0]) <= GLM_FLT_EPSILON
         && fabsf(a[1] - b[1]) <= GLM_FLT_EPSILON
         && fabsf(a[2] - b[2]) <= GLM_FLT_EPSILON
         && fabsf(a[3] - b[3]) <= GLM_FLT_EPSILON;
}

/*!
 * @brief max value of vector
 *
 * @param v vector
 */
CGLM_INLINE
float
glm_vec4_max(vec4 v) {
  float max;

  max = glm_vec3_max(v);
  if (v[3] > max)
    max = v[3];

  return max;
}

/*!
 * @brief min value of vector
 *
 * @param v vector
 */
CGLM_INLINE
float
glm_vec4_min(vec4 v) {
  float min;

  min = glm_vec3_min(v);
  if (v[3] < min)
    min = v[3];

  return min;
}

/*!
 * @brief check if one of items is NaN (not a number)
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
CGLM_INLINE
bool
glm_vec4_isnan(vec4 v) {
  return isnan(v[0]) || isnan(v[1]) || isnan(v[2]) || isnan(v[3]);
}

/*!
 * @brief check if one of items is INFINITY
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
CGLM_INLINE
bool
glm_vec4_isinf(vec4 v) {
  return isinf(v[0]) || isinf(v[1]) || isinf(v[2]) || isinf(v[3]);
}

/*!
 * @brief check if all items are valid number
 *        you should only use this in DEBUG mode or very critical asserts
 *
 * @param[in] v vector
 */
CGLM_INLINE
bool
glm_vec4_isvalid(vec4 v) {
  return !glm_vec4_isnan(v) && !glm_vec4_isinf(v);
}

/*!
 * @brief get sign of 32 bit float as +1, -1, 0
 *
 * Important: It returns 0 for zero/NaN input
 *
 * @param v vector
 */
CGLM_INLINE
void
glm_vec4_sign(vec4 v, vec4 dest) {
#if defined( __SSE2__ ) || defined( __SSE2__ )
  __m128 x0, x1, x2, x3, x4;

  x0 = glmm_load(v);
  x1 = _mm_set_ps(0.0f, 0.0f, 1.0f, -1.0f);
  x2 = glmm_shuff1x(x1, 2);

  x3 = _mm_and_ps(_mm_cmpgt_ps(x0, x2), glmm_shuff1x(x1, 1));
  x4 = _mm_and_ps(_mm_cmplt_ps(x0, x2), glmm_shuff1x(x1, 0));

  glmm_store(dest, _mm_or_ps(x3, x4));
#else
  dest[0] = glm_signf(v[0]);
  dest[1] = glm_signf(v[1]);
  dest[2] = glm_signf(v[2]);
  dest[3] = glm_signf(v[3]);
#endif
}

/*!
 * @brief absolute value of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec4_abs(vec4 v, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, glmm_abs(glmm_load(v)));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vabsq_f32(vld1q_f32(v)));
#else
  dest[0] = fabsf(v[0]);
  dest[1] = fabsf(v[1]);
  dest[2] = fabsf(v[2]);
  dest[3] = fabsf(v[3]);
#endif
}

/*!
 * @brief fractional part of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec4_fract(vec4 v, vec4 dest) {
  dest[0] = fminf(v[0] - floorf(v[0]), 0.999999940395355224609375f);
  dest[1] = fminf(v[1] - floorf(v[1]), 0.999999940395355224609375f);
  dest[2] = fminf(v[2] - floorf(v[2]), 0.999999940395355224609375f);
  dest[3] = fminf(v[3] - floorf(v[3]), 0.999999940395355224609375f);
}

/*!
 * @brief vector reduction by summation
 * @warning could overflow
 *
 * @param[in]   v    vector
 * @return      sum of all vector's elements
 */
CGLM_INLINE
float
glm_vec4_hadd(vec4 v) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  return glmm_hadd(glmm_load(v));
#else
  return v[0] + v[1] + v[2] + v[3];
#endif
}

/*!
 * @brief square root of each vector item
 *
 * @param[in]  v    vector
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec4_sqrt(vec4 v, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_sqrt_ps(glmm_load(v)));
#else
  dest[0] = sqrtf(v[0]);
  dest[1] = sqrtf(v[1]);
  dest[2] = sqrtf(v[2]);
  dest[3] = sqrtf(v[3]);
#endif
}

#endif /* cglm_vec4_ext_h */


/* DEPRECATED! functions */
#define glm_vec4_dup3(v, dest)         glm_vec4_copy3(v, dest)
#define glm_vec4_dup(v, dest)          glm_vec4_copy(v, dest)
#define glm_vec4_flipsign(v)           glm_vec4_negate(v)
#define glm_vec4_flipsign_to(v, dest)  glm_vec4_negate_to(v, dest)
#define glm_vec4_inv(v)                glm_vec4_negate(v)
#define glm_vec4_inv_to(v, dest)       glm_vec4_negate_to(v, dest)
#define glm_vec4_mulv(a, b, d)         glm_vec4_mul(a, b, d)

#define GLM_VEC4_ONE_INIT   {1.0f, 1.0f, 1.0f, 1.0f}
#define GLM_VEC4_BLACK_INIT {0.0f, 0.0f, 0.0f, 1.0f}
#define GLM_VEC4_ZERO_INIT  {0.0f, 0.0f, 0.0f, 0.0f}

#define GLM_VEC4_ONE        ((vec4)GLM_VEC4_ONE_INIT)
#define GLM_VEC4_BLACK      ((vec4)GLM_VEC4_BLACK_INIT)
#define GLM_VEC4_ZERO       ((vec4)GLM_VEC4_ZERO_INIT)

#define GLM_XXXX GLM_SHUFFLE4(0, 0, 0, 0)
#define GLM_YYYY GLM_SHUFFLE4(1, 1, 1, 1)
#define GLM_ZZZZ GLM_SHUFFLE4(2, 2, 2, 2)
#define GLM_WWWW GLM_SHUFFLE4(3, 3, 3, 3)
#define GLM_WZYX GLM_SHUFFLE4(0, 1, 2, 3)

/*!
 * @brief init vec4 using vec3
 *
 * @param[in]  v3   vector3
 * @param[in]  last last item
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec4(vec3 v3, float last, vec4 dest) {
  dest[0] = v3[0];
  dest[1] = v3[1];
  dest[2] = v3[2];
  dest[3] = last;
}

/*!
 * @brief copy first 3 members of [a] to [dest]
 *
 * @param[in]  a    source
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec4_copy3(vec4 a, vec3 dest) {
  dest[0] = a[0];
  dest[1] = a[1];
  dest[2] = a[2];
}

/*!
 * @brief copy all members of [a] to [dest]
 *
 * @param[in]  v    source
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec4_copy(vec4 v, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, glmm_load(v));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vld1q_f32(v));
#else
  dest[0] = v[0];
  dest[1] = v[1];
  dest[2] = v[2];
  dest[3] = v[3];
#endif
}

/*!
 * @brief copy all members of [a] to [dest]
 *
 * alignment is not required
 *
 * @param[in]  v    source
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec4_ucopy(vec4 v, vec4 dest) {
  dest[0] = v[0];
  dest[1] = v[1];
  dest[2] = v[2];
  dest[3] = v[3];
}

/*!
 * @brief make vector zero
 *
 * @param[in, out]  v vector
 */
CGLM_INLINE
void
glm_vec4_zero(vec4 v) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(v, _mm_setzero_ps());
#elif defined(CGLM_NEON_FP)
  vst1q_f32(v, vdupq_n_f32(0.0f));
#else
  v[0] = 0.0f;
  v[1] = 0.0f;
  v[2] = 0.0f;
  v[3] = 0.0f;
#endif
}

/*!
 * @brief make vector one
 *
 * @param[in, out]  v vector
 */
CGLM_INLINE
void
glm_vec4_one(vec4 v) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(v, _mm_set1_ps(1.0f));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(v, vdupq_n_f32(1.0f));
#else
  v[0] = 1.0f;
  v[1] = 1.0f;
  v[2] = 1.0f;
  v[3] = 1.0f;
#endif
}

/*!
 * @brief vec4 dot product
 *
 * @param[in] a vector1
 * @param[in] b vector2
 *
 * @return dot product
 */
CGLM_INLINE
float
glm_vec4_dot(vec4 a, vec4 b) {
#if defined(CGLM_SIMD)
  return glmm_dot(glmm_load(a), glmm_load(b));
#else
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
#endif
}

/*!
 * @brief norm * norm (magnitude) of vec
 *
 * we can use this func instead of calling norm * norm, because it would call
 * sqrtf fuction twice but with this func we can avoid func call, maybe this is
 * not good name for this func
 *
 * @param[in] v vec4
 *
 * @return norm * norm
 */
CGLM_INLINE
float
glm_vec4_norm2(vec4 v) {
  return glm_vec4_dot(v, v);
}

/*!
 * @brief euclidean norm (magnitude), also called L2 norm
 *        this will give magnitude of vector in euclidean space
 *
 * @param[in] v vector
 *
 * @return norm
 */
CGLM_INLINE
float
glm_vec4_norm(vec4 v) {
#if defined(CGLM_SIMD)
  return glmm_norm(glmm_load(v));
#else
  return sqrtf(glm_vec4_dot(v, v));
#endif
}

/*!
 * @brief L1 norm of vec4
 * Also known as Manhattan Distance or Taxicab norm.
 * L1 Norm is the sum of the magnitudes of the vectors in a space.
 * It is calculated as the sum of the absolute values of the vector components.
 * In this norm, all the components of the vector are weighted equally.
 *
 * This computes:
 * L1 norm = |v[0]| + |v[1]| + |v[2]| + |v[3]|
 *
 * @param[in] v vector
 *
 * @return L1 norm
 */
CGLM_INLINE
float
glm_vec4_norm_one(vec4 v) {
#if defined(CGLM_SIMD)
  return glmm_norm_one(glmm_load(v));
#else
  vec4 t;
  glm_vec4_abs(v, t);
  return glm_vec4_hadd(t);
#endif
}

/*!
 * @brief infinity norm of vec4
 * Also known as Maximum norm.
 * Infinity Norm is the largest magnitude among each element of a vector.
 * It is calculated as the maximum of the absolute values of the vector components.
 *
 * This computes:
 * inf norm = max(|v[0]|, |v[1]|, |v[2]|, |v[3]|)
 *
 * @param[in] v vector
 *
 * @return infinity norm
 */
CGLM_INLINE
float
glm_vec4_norm_inf(vec4 v) {
#if defined(CGLM_SIMD)
  return glmm_norm_inf(glmm_load(v));
#else
  vec4 t;
  glm_vec4_abs(v, t);
  return glm_vec4_max(t);
#endif
}

/*!
 * @brief add b vector to a vector store result in dest
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec4_add(vec4 a, vec4 b, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_add_ps(glmm_load(a), glmm_load(b)));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vaddq_f32(vld1q_f32(a), vld1q_f32(b)));
#else
  dest[0] = a[0] + b[0];
  dest[1] = a[1] + b[1];
  dest[2] = a[2] + b[2];
  dest[3] = a[3] + b[3];
#endif
}

/*!
 * @brief add scalar to v vector store result in dest (d = v + vec(s))
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec4_adds(vec4 v, float s, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_add_ps(glmm_load(v), _mm_set1_ps(s)));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vaddq_f32(vld1q_f32(v), vdupq_n_f32(s)));
#else
  dest[0] = v[0] + s;
  dest[1] = v[1] + s;
  dest[2] = v[2] + s;
  dest[3] = v[3] + s;
#endif
}

/*!
 * @brief subtract b vector from a vector store result in dest (d = a - b)
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec4_sub(vec4 a, vec4 b, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_sub_ps(glmm_load(a), glmm_load(b)));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vsubq_f32(vld1q_f32(a), vld1q_f32(b)));
#else
  dest[0] = a[0] - b[0];
  dest[1] = a[1] - b[1];
  dest[2] = a[2] - b[2];
  dest[3] = a[3] - b[3];
#endif
}

/*!
 * @brief subtract scalar from v vector store result in dest (d = v - vec(s))
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec4_subs(vec4 v, float s, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_sub_ps(glmm_load(v), _mm_set1_ps(s)));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vsubq_f32(vld1q_f32(v), vdupq_n_f32(s)));
#else
  dest[0] = v[0] - s;
  dest[1] = v[1] - s;
  dest[2] = v[2] - s;
  dest[3] = v[3] - s;
#endif
}

/*!
 * @brief multiply two vector (component-wise multiplication)
 *
 * @param a    vector1
 * @param b    vector2
 * @param dest dest = (a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3])
 */
CGLM_INLINE
void
glm_vec4_mul(vec4 a, vec4 b, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_mul_ps(glmm_load(a), glmm_load(b)));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vmulq_f32(vld1q_f32(a), vld1q_f32(b)));
#else
  dest[0] = a[0] * b[0];
  dest[1] = a[1] * b[1];
  dest[2] = a[2] * b[2];
  dest[3] = a[3] * b[3];
#endif
}

/*!
 * @brief multiply/scale vec4 vector with scalar: result = v * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec4_scale(vec4 v, float s, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_mul_ps(glmm_load(v), _mm_set1_ps(s)));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vmulq_f32(vld1q_f32(v), vdupq_n_f32(s)));
#else
  dest[0] = v[0] * s;
  dest[1] = v[1] * s;
  dest[2] = v[2] * s;
  dest[3] = v[3] * s;
#endif
}

/*!
 * @brief make vec4 vector scale as specified: result = unit(v) * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec4_scale_as(vec4 v, float s, vec4 dest) {
  float norm;
  norm = glm_vec4_norm(v);

  if (norm == 0.0f) {
    glm_vec4_zero(dest);
    return;
  }

  glm_vec4_scale(v, s / norm, dest);
}

/*!
 * @brief div vector with another component-wise division: d = a / b
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest result = (a[0]/b[0], a[1]/b[1], a[2]/b[2], a[3]/b[3])
 */
CGLM_INLINE
void
glm_vec4_div(vec4 a, vec4 b, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_div_ps(glmm_load(a), glmm_load(b)));
#else
  dest[0] = a[0] / b[0];
  dest[1] = a[1] / b[1];
  dest[2] = a[2] / b[2];
  dest[3] = a[3] / b[3];
#endif
}

/*!
 * @brief div vec4 vector with scalar: d = v / s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec4_divs(vec4 v, float s, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_div_ps(glmm_load(v), _mm_set1_ps(s)));
#else
  glm_vec4_scale(v, 1.0f / s, dest);
#endif
}

/*!
 * @brief add two vectors and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a + b)
 */
CGLM_INLINE
void
glm_vec4_addadd(vec4 a, vec4 b, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_add_ps(glmm_load(dest),
                              _mm_add_ps(glmm_load(a),
                                         glmm_load(b))));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vaddq_f32(vld1q_f32(dest),
                            vaddq_f32(vld1q_f32(a),
                                      vld1q_f32(b))));
#else
  dest[0] += a[0] + b[0];
  dest[1] += a[1] + b[1];
  dest[2] += a[2] + b[2];
  dest[3] += a[3] + b[3];
#endif
}

/*!
 * @brief sub two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a - b)
 */
CGLM_INLINE
void
glm_vec4_subadd(vec4 a, vec4 b, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_add_ps(glmm_load(dest),
                              _mm_sub_ps(glmm_load(a),
                                         glmm_load(b))));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vaddq_f32(vld1q_f32(dest),
                            vsubq_f32(vld1q_f32(a),
                                      vld1q_f32(b))));
#else
  dest[0] += a[0] - b[0];
  dest[1] += a[1] - b[1];
  dest[2] += a[2] - b[2];
  dest[3] += a[3] - b[3];
#endif
}

/*!
 * @brief mul two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a * b)
 */
CGLM_INLINE
void
glm_vec4_muladd(vec4 a, vec4 b, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_add_ps(glmm_load(dest),
                              _mm_mul_ps(glmm_load(a),
                                         glmm_load(b))));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vaddq_f32(vld1q_f32(dest),
                            vmulq_f32(vld1q_f32(a),
                                      vld1q_f32(b))));
#else
  dest[0] += a[0] * b[0];
  dest[1] += a[1] * b[1];
  dest[2] += a[2] * b[2];
  dest[3] += a[3] * b[3];
#endif
}

/*!
 * @brief mul vector with scalar and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest += (a * b)
 */
CGLM_INLINE
void
glm_vec4_muladds(vec4 a, float s, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_add_ps(glmm_load(dest),
                              _mm_mul_ps(glmm_load(a),
                                         _mm_set1_ps(s))));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vaddq_f32(vld1q_f32(dest),
                            vmulq_f32(vld1q_f32(a),
                                      vdupq_n_f32(s))));
#else
  dest[0] += a[0] * s;
  dest[1] += a[1] * s;
  dest[2] += a[2] * s;
  dest[3] += a[3] * s;
#endif
}

/*!
 * @brief add max of two vector to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += max(a, b)
 */
CGLM_INLINE
void
glm_vec4_maxadd(vec4 a, vec4 b, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_add_ps(glmm_load(dest),
                              _mm_max_ps(glmm_load(a),
                                         glmm_load(b))));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vaddq_f32(vld1q_f32(dest),
                            vmaxq_f32(vld1q_f32(a),
                                      vld1q_f32(b))));
#else
  dest[0] += glm_max(a[0], b[0]);
  dest[1] += glm_max(a[1], b[1]);
  dest[2] += glm_max(a[2], b[2]);
  dest[3] += glm_max(a[3], b[3]);
#endif
}

/*!
 * @brief add min of two vector to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += min(a, b)
 */
CGLM_INLINE
void
glm_vec4_minadd(vec4 a, vec4 b, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_add_ps(glmm_load(dest),
                              _mm_min_ps(glmm_load(a),
                                         glmm_load(b))));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vaddq_f32(vld1q_f32(dest),
                            vminq_f32(vld1q_f32(a),
                                      vld1q_f32(b))));
#else
  dest[0] += glm_min(a[0], b[0]);
  dest[1] += glm_min(a[1], b[1]);
  dest[2] += glm_min(a[2], b[2]);
  dest[3] += glm_min(a[3], b[3]);
#endif
}

/*!
 * @brief negate vector components and store result in dest
 *
 * @param[in]  v     vector
 * @param[out] dest  result vector
 */
CGLM_INLINE
void
glm_vec4_negate_to(vec4 v, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_xor_ps(glmm_load(v), _mm_set1_ps(-0.0f)));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vnegq_f32(vld1q_f32(v)));
#else
  dest[0] = -v[0];
  dest[1] = -v[1];
  dest[2] = -v[2];
  dest[3] = -v[3];
#endif
}

/*!
 * @brief flip sign of all vec4 members
 *
 * @param[in, out]  v  vector
 */
CGLM_INLINE
void
glm_vec4_negate(vec4 v) {
  glm_vec4_negate_to(v, v);
}

/*!
 * @brief normalize vec4 to dest
 *
 * @param[in]  v    source
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec4_normalize_to(vec4 v, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  __m128 xdot, x0;
  float  dot;

  x0   = glmm_load(v);
  xdot = glmm_vdot(x0, x0);
  dot  = _mm_cvtss_f32(xdot);

  if (dot == 0.0f) {
    glmm_store(dest, _mm_setzero_ps());
    return;
  }

  glmm_store(dest, _mm_div_ps(x0, _mm_sqrt_ps(xdot)));
#else
  float norm;

  norm = glm_vec4_norm(v);

  if (norm == 0.0f) {
    glm_vec4_zero(dest);
    return;
  }

  glm_vec4_scale(v, 1.0f / norm, dest);
#endif
}

/*!
 * @brief normalize vec4 and store result in same vec
 *
 * @param[in, out] v vector
 */
CGLM_INLINE
void
glm_vec4_normalize(vec4 v) {
  glm_vec4_normalize_to(v, v);
}

/**
 * @brief distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return returns distance
 */
CGLM_INLINE
float
glm_vec4_distance(vec4 a, vec4 b) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  return glmm_norm(_mm_sub_ps(glmm_load(a), glmm_load(b)));
#elif defined(CGLM_NEON_FP)
  return glmm_norm(vsubq_f32(glmm_load(a), glmm_load(b)));
#else
  return sqrtf(glm_pow2(a[0] - b[0])
             + glm_pow2(a[1] - b[1])
             + glm_pow2(a[2] - b[2])
             + glm_pow2(a[3] - b[3]));
#endif
}

/**
 * @brief squared distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return returns squared distance
 */
CGLM_INLINE
float
glm_vec4_distance2(vec4 a, vec4 b) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  return glmm_norm2(_mm_sub_ps(glmm_load(a), glmm_load(b)));
#elif defined(CGLM_NEON_FP)
  return glmm_norm2(vsubq_f32(glmm_load(a), glmm_load(b)));
#else
  return glm_pow2(a[0] - b[0])
       + glm_pow2(a[1] - b[1])
       + glm_pow2(a[2] - b[2])
       + glm_pow2(a[3] - b[3]);
#endif
}

/*!
 * @brief max values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec4_maxv(vec4 a, vec4 b, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_max_ps(glmm_load(a), glmm_load(b)));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vmaxq_f32(vld1q_f32(a), vld1q_f32(b)));
#else
  dest[0] = glm_max(a[0], b[0]);
  dest[1] = glm_max(a[1], b[1]);
  dest[2] = glm_max(a[2], b[2]);
  dest[3] = glm_max(a[3], b[3]);
#endif
}

/*!
 * @brief min values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec4_minv(vec4 a, vec4 b, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest, _mm_min_ps(glmm_load(a), glmm_load(b)));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest, vminq_f32(vld1q_f32(a), vld1q_f32(b)));
#else
  dest[0] = glm_min(a[0], b[0]);
  dest[1] = glm_min(a[1], b[1]);
  dest[2] = glm_min(a[2], b[2]);
  dest[3] = glm_min(a[3], b[3]);
#endif
}

/*!
 * @brief clamp vector's individual members between min and max values
 *
 * @param[in, out]  v      vector
 * @param[in]       minVal minimum value
 * @param[in]       maxVal maximum value
 */
CGLM_INLINE
void
glm_vec4_clamp(vec4 v, float minVal, float maxVal) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(v, _mm_min_ps(_mm_max_ps(glmm_load(v), _mm_set1_ps(minVal)),
                           _mm_set1_ps(maxVal)));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(v, vminq_f32(vmaxq_f32(vld1q_f32(v), vdupq_n_f32(minVal)),
                         vdupq_n_f32(maxVal)));
#else
  v[0] = glm_clamp(v[0], minVal, maxVal);
  v[1] = glm_clamp(v[1], minVal, maxVal);
  v[2] = glm_clamp(v[2], minVal, maxVal);
  v[3] = glm_clamp(v[3], minVal, maxVal);
#endif
}

/*!
 * @brief linear interpolation between two vectors
 *
 * formula:  from + t * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount)
 * @param[out]  dest destination
 */
CGLM_INLINE
void
glm_vec4_lerp(vec4 from, vec4 to, float t, vec4 dest) {
  vec4 s, v;

  /* from + s * (to - from) */
  glm_vec4_broadcast(t, s);
  glm_vec4_sub(to, from, v);
  glm_vec4_mul(s, v, v);
  glm_vec4_add(from, v, dest);
}

/*!
 * @brief linear interpolation between two vectors (clamped)
 *
 * formula:  from + t * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount) clamped between 0 and 1
 * @param[out]  dest destination
 */
CGLM_INLINE
void
glm_vec4_lerpc(vec4 from, vec4 to, float t, vec4 dest) {
  glm_vec4_lerp(from, to, glm_clamp_zo(t), dest);
}

/*!
 * @brief linear interpolation between two vectors
 *
 * formula:  from + t * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount)
 * @param[out]  dest destination
 */
CGLM_INLINE
void
glm_vec4_mix(vec4 from, vec4 to, float t, vec4 dest) {
  glm_vec4_lerp(from, to, t, dest);
}

/*!
 * @brief linear interpolation between two vectors (clamped)
 *
 * formula:  from + t * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount) clamped between 0 and 1
 * @param[out]  dest destination
 */
CGLM_INLINE
void
glm_vec4_mixc(vec4 from, vec4 to, float t, vec4 dest) {
  glm_vec4_lerpc(from, to, t, dest);
}

/*!
 * @brief threshold function (unidimensional)
 *
 * @param[in]   edge    threshold
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
CGLM_INLINE
void
glm_vec4_step_uni(float edge, vec4 x, vec4 dest) {
  dest[0] = glm_step(edge, x[0]);
  dest[1] = glm_step(edge, x[1]);
  dest[2] = glm_step(edge, x[2]);
  dest[3] = glm_step(edge, x[3]);
}

/*!
 * @brief threshold function
 *
 * @param[in]   edge    threshold
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
CGLM_INLINE
void
glm_vec4_step(vec4 edge, vec4 x, vec4 dest) {
  dest[0] = glm_step(edge[0], x[0]);
  dest[1] = glm_step(edge[1], x[1]);
  dest[2] = glm_step(edge[2], x[2]);
  dest[3] = glm_step(edge[3], x[3]);
}

/*!
 * @brief threshold function with a smooth transition (unidimensional)
 *
 * @param[in]   edge0   low threshold
 * @param[in]   edge1   high threshold
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
CGLM_INLINE
void
glm_vec4_smoothstep_uni(float edge0, float edge1, vec4 x, vec4 dest) {
  dest[0] = glm_smoothstep(edge0, edge1, x[0]);
  dest[1] = glm_smoothstep(edge0, edge1, x[1]);
  dest[2] = glm_smoothstep(edge0, edge1, x[2]);
  dest[3] = glm_smoothstep(edge0, edge1, x[3]);
}

/*!
 * @brief threshold function with a smooth transition
 *
 * @param[in]   edge0   low threshold
 * @param[in]   edge1   high threshold
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
CGLM_INLINE
void
glm_vec4_smoothstep(vec4 edge0, vec4 edge1, vec4 x, vec4 dest) {
  dest[0] = glm_smoothstep(edge0[0], edge1[0], x[0]);
  dest[1] = glm_smoothstep(edge0[1], edge1[1], x[1]);
  dest[2] = glm_smoothstep(edge0[2], edge1[2], x[2]);
  dest[3] = glm_smoothstep(edge0[3], edge1[3], x[3]);
}

/*!
 * @brief smooth Hermite interpolation between two vectors
 *
 * formula:  t^2 * (3 - 2*t)
 *
 * @param[in]   from    from value
 * @param[in]   to      to value
 * @param[in]   t       interpolant (amount)
 * @param[out]  dest    destination
 */
CGLM_INLINE
void
glm_vec4_smoothinterp(vec4 from, vec4 to, float t, vec4 dest) {
  vec4 s, v;
    
  /* from + smoothstep * (to - from) */
  glm_vec4_broadcast(glm_smooth(t), s);
  glm_vec4_sub(to, from, v);
  glm_vec4_mul(s, v, v);
  glm_vec4_add(from, v, dest);
}

/*!
 * @brief smooth Hermite interpolation between two vectors (clamped)
 *
 * formula:  t^2 * (3 - 2*t)
 *
 * @param[in]   from    from value
 * @param[in]   to      to value
 * @param[in]   t       interpolant (amount) clamped between 0 and 1
 * @param[out]  dest    destination
 */
CGLM_INLINE
void
glm_vec4_smoothinterpc(vec4 from, vec4 to, float t, vec4 dest) {
  glm_vec4_smoothinterp(from, to, glm_clamp_zo(t), dest);
}

/*!
 * @brief helper to fill vec4 as [S^3, S^2, S, 1]
 *
 * @param[in]   s    parameter
 * @param[out]  dest destination
 */
CGLM_INLINE
void
glm_vec4_cubic(float s, vec4 dest) {
  float ss;

  ss = s * s;

  dest[0] = ss * s;
  dest[1] = ss;
  dest[2] = s;
  dest[3] = 1.0f;
}

/*!
 * @brief swizzle vector components
 *
 * you can use existin masks e.g. GLM_XXXX, GLM_WZYX
 *
 * @param[in]  v    source
 * @param[in]  mask mask
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec4_swizzle(vec4 v, int mask, vec4 dest) {
  vec4 t;

  t[0] = v[(mask & (3 << 0))];
  t[1] = v[(mask & (3 << 2)) >> 2];
  t[2] = v[(mask & (3 << 4)) >> 4];
  t[3] = v[(mask & (3 << 6)) >> 6];

  glm_vec4_copy(t, dest);
}

#endif /* cglm_vec4_h */


/* DEPRECATED! use _copy, _ucopy versions */
#define glm_vec3_dup(v, dest)         glm_vec3_copy(v, dest)
#define glm_vec3_flipsign(v)          glm_vec3_negate(v)
#define glm_vec3_flipsign_to(v, dest) glm_vec3_negate_to(v, dest)
#define glm_vec3_inv(v)               glm_vec3_negate(v)
#define glm_vec3_inv_to(v, dest)      glm_vec3_negate_to(v, dest)
#define glm_vec3_mulv(a, b, d)        glm_vec3_mul(a, b, d)

#define GLM_VEC3_ONE_INIT   {1.0f, 1.0f, 1.0f}
#define GLM_VEC3_ZERO_INIT  {0.0f, 0.0f, 0.0f}

#define GLM_VEC3_ONE  ((vec3)GLM_VEC3_ONE_INIT)
#define GLM_VEC3_ZERO ((vec3)GLM_VEC3_ZERO_INIT)

#define GLM_YUP       ((vec3){0.0f,  1.0f,  0.0f})
#define GLM_ZUP       ((vec3){0.0f,  0.0f,  1.0f})
#define GLM_XUP       ((vec3){1.0f,  0.0f,  0.0f})
#define GLM_FORWARD   ((vec3){0.0f,  0.0f, -1.0f})

#define GLM_XXX GLM_SHUFFLE3(0, 0, 0)
#define GLM_YYY GLM_SHUFFLE3(1, 1, 1)
#define GLM_ZZZ GLM_SHUFFLE3(2, 2, 2)
#define GLM_ZYX GLM_SHUFFLE3(0, 1, 2)

/*!
 * @brief init vec3 using vec4
 *
 * @param[in]  v4   vector4
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec3(vec4 v4, vec3 dest) {
  dest[0] = v4[0];
  dest[1] = v4[1];
  dest[2] = v4[2];
}

/*!
 * @brief copy all members of [a] to [dest]
 *
 * @param[in]  a    source
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec3_copy(vec3 a, vec3 dest) {
  dest[0] = a[0];
  dest[1] = a[1];
  dest[2] = a[2];
}

/*!
 * @brief make vector zero
 *
 * @param[in, out]  v vector
 */
CGLM_INLINE
void
glm_vec3_zero(vec3 v) {
  v[0] = v[1] = v[2] = 0.0f;
}

/*!
 * @brief make vector one
 *
 * @param[in, out]  v vector
 */
CGLM_INLINE
void
glm_vec3_one(vec3 v) {
  v[0] = v[1] = v[2] = 1.0f;
}

/*!
 * @brief vec3 dot product
 *
 * @param[in] a vector1
 * @param[in] b vector2
 *
 * @return dot product
 */
CGLM_INLINE
float
glm_vec3_dot(vec3 a, vec3 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/*!
 * @brief norm * norm (magnitude) of vec
 *
 * we can use this func instead of calling norm * norm, because it would call
 * sqrtf fuction twice but with this func we can avoid func call, maybe this is
 * not good name for this func
 *
 * @param[in] v vector
 *
 * @return norm * norm
 */
CGLM_INLINE
float
glm_vec3_norm2(vec3 v) {
  return glm_vec3_dot(v, v);
}

/*!
 * @brief euclidean norm (magnitude), also called L2 norm
 *        this will give magnitude of vector in euclidean space
 *
 * @param[in] v vector
 *
 * @return norm
 */
CGLM_INLINE
float
glm_vec3_norm(vec3 v) {
  return sqrtf(glm_vec3_norm2(v));
}

/*!
 * @brief L1 norm of vec3
 * Also known as Manhattan Distance or Taxicab norm.
 * L1 Norm is the sum of the magnitudes of the vectors in a space.
 * It is calculated as the sum of the absolute values of the vector components.
 * In this norm, all the components of the vector are weighted equally.
 *
 * This computes:
 * R = |v[0]| + |v[1]| + |v[2]|
 *
 * @param[in] v vector
 *
 * @return L1 norm
 */
CGLM_INLINE
float
glm_vec3_norm_one(vec3 v) {
  vec3 t;
  glm_vec3_abs(v, t);
  return glm_vec3_hadd(t);
}

/*!
 * @brief infinity norm of vec3
 * Also known as Maximum norm.
 * Infinity Norm is the largest magnitude among each element of a vector.
 * It is calculated as the maximum of the absolute values of the vector components.
 *
 * This computes:
 * inf norm = max(|v[0]|, |v[1]|, |v[2]|)
 *
 * @param[in] v vector
 *
 * @return infinity norm
 */
CGLM_INLINE
float
glm_vec3_norm_inf(vec3 v) {
  vec3 t;
  glm_vec3_abs(v, t);
  return glm_vec3_max(t);
}

/*!
 * @brief add a vector to b vector store result in dest
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec3_add(vec3 a, vec3 b, vec3 dest) {
  dest[0] = a[0] + b[0];
  dest[1] = a[1] + b[1];
  dest[2] = a[2] + b[2];
}

/*!
 * @brief add scalar to v vector store result in dest (d = v + s)
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec3_adds(vec3 v, float s, vec3 dest) {
  dest[0] = v[0] + s;
  dest[1] = v[1] + s;
  dest[2] = v[2] + s;
}

/*!
 * @brief subtract b vector from a vector store result in dest
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec3_sub(vec3 a, vec3 b, vec3 dest) {
  dest[0] = a[0] - b[0];
  dest[1] = a[1] - b[1];
  dest[2] = a[2] - b[2];
}

/*!
 * @brief subtract scalar from v vector store result in dest (d = v - s)
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec3_subs(vec3 v, float s, vec3 dest) {
  dest[0] = v[0] - s;
  dest[1] = v[1] - s;
  dest[2] = v[2] - s;
}

/*!
 * @brief multiply two vector (component-wise multiplication)
 *
 * @param a    vector1
 * @param b    vector2
 * @param dest v3 = (a[0] * b[0], a[1] * b[1], a[2] * b[2])
 */
CGLM_INLINE
void
glm_vec3_mul(vec3 a, vec3 b, vec3 dest) {
  dest[0] = a[0] * b[0];
  dest[1] = a[1] * b[1];
  dest[2] = a[2] * b[2];
}

/*!
 * @brief multiply/scale vec3 vector with scalar: result = v * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec3_scale(vec3 v, float s, vec3 dest) {
  dest[0] = v[0] * s;
  dest[1] = v[1] * s;
  dest[2] = v[2] * s;
}

/*!
 * @brief make vec3 vector scale as specified: result = unit(v) * s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest destination vector
 */
CGLM_INLINE
void
glm_vec3_scale_as(vec3 v, float s, vec3 dest) {
  float norm;
  norm = glm_vec3_norm(v);

  if (norm == 0.0f) {
    glm_vec3_zero(dest);
    return;
  }

  glm_vec3_scale(v, s / norm, dest);
}

/*!
 * @brief div vector with another component-wise division: d = a / b
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest result = (a[0]/b[0], a[1]/b[1], a[2]/b[2])
 */
CGLM_INLINE
void
glm_vec3_div(vec3 a, vec3 b, vec3 dest) {
  dest[0] = a[0] / b[0];
  dest[1] = a[1] / b[1];
  dest[2] = a[2] / b[2];
}

/*!
 * @brief div vector with scalar: d = v / s
 *
 * @param[in]  v    vector
 * @param[in]  s    scalar
 * @param[out] dest result = (a[0]/s, a[1]/s, a[2]/s)
 */
CGLM_INLINE
void
glm_vec3_divs(vec3 v, float s, vec3 dest) {
  dest[0] = v[0] / s;
  dest[1] = v[1] / s;
  dest[2] = v[2] / s;
}

/*!
 * @brief add two vectors and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a + b)
 */
CGLM_INLINE
void
glm_vec3_addadd(vec3 a, vec3 b, vec3 dest) {
  dest[0] += a[0] + b[0];
  dest[1] += a[1] + b[1];
  dest[2] += a[2] + b[2];
}

/*!
 * @brief sub two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a + b)
 */
CGLM_INLINE
void
glm_vec3_subadd(vec3 a, vec3 b, vec3 dest) {
  dest[0] += a[0] - b[0];
  dest[1] += a[1] - b[1];
  dest[2] += a[2] - b[2];
}

/*!
 * @brief mul two vectors and add result to dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += (a * b)
 */
CGLM_INLINE
void
glm_vec3_muladd(vec3 a, vec3 b, vec3 dest) {
  dest[0] += a[0] * b[0];
  dest[1] += a[1] * b[1];
  dest[2] += a[2] * b[2];
}

/*!
 * @brief mul vector with scalar and add result to sum
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector
 * @param[in]  s    scalar
 * @param[out] dest dest += (a * b)
 */
CGLM_INLINE
void
glm_vec3_muladds(vec3 a, float s, vec3 dest) {
  dest[0] += a[0] * s;
  dest[1] += a[1] * s;
  dest[2] += a[2] * s;
}

/*!
 * @brief add max of two vector to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += max(a, b)
 */
CGLM_INLINE
void
glm_vec3_maxadd(vec3 a, vec3 b, vec3 dest) {
  dest[0] += glm_max(a[0], b[0]);
  dest[1] += glm_max(a[1], b[1]);
  dest[2] += glm_max(a[2], b[2]);
}

/*!
 * @brief add min of two vector to result/dest
 *
 * it applies += operator so dest must be initialized
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest dest += min(a, b)
 */
CGLM_INLINE
void
glm_vec3_minadd(vec3 a, vec3 b, vec3 dest) {
  dest[0] += glm_min(a[0], b[0]);
  dest[1] += glm_min(a[1], b[1]);
  dest[2] += glm_min(a[2], b[2]);
}

/*!
 * @brief negate vector components and store result in dest
 *
 * @param[in]   v     vector
 * @param[out]  dest  result vector
 */
CGLM_INLINE
void
glm_vec3_negate_to(vec3 v, vec3 dest) {
  dest[0] = -v[0];
  dest[1] = -v[1];
  dest[2] = -v[2];
}

/*!
 * @brief negate vector components
 *
 * @param[in, out]  v  vector
 */
CGLM_INLINE
void
glm_vec3_negate(vec3 v) {
  glm_vec3_negate_to(v, v);
}

/*!
 * @brief normalize vec3 and store result in same vec
 *
 * @param[in, out] v vector
 */
CGLM_INLINE
void
glm_vec3_normalize(vec3 v) {
  float norm;

  norm = glm_vec3_norm(v);

  if (norm == 0.0f) {
    v[0] = v[1] = v[2] = 0.0f;
    return;
  }

  glm_vec3_scale(v, 1.0f / norm, v);
}

/*!
 * @brief normalize vec3 to dest
 *
 * @param[in]  v    source
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec3_normalize_to(vec3 v, vec3 dest) {
  float norm;

  norm = glm_vec3_norm(v);

  if (norm == 0.0f) {
    glm_vec3_zero(dest);
    return;
  }

  glm_vec3_scale(v, 1.0f / norm, dest);
}

/*!
 * @brief cross product of two vector (RH)
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec3_cross(vec3 a, vec3 b, vec3 dest) {
  /* (u2.v3 - u3.v2, u3.v1 - u1.v3, u1.v2 - u2.v1) */
  dest[0] = a[1] * b[2] - a[2] * b[1];
  dest[1] = a[2] * b[0] - a[0] * b[2];
  dest[2] = a[0] * b[1] - a[1] * b[0];
}

/*!
 * @brief cross product of two vector (RH) and normalize the result
 *
 * @param[in]  a    vector 1
 * @param[in]  b    vector 2
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec3_crossn(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_cross(a, b, dest);
  glm_vec3_normalize(dest);
}

/*!
 * @brief angle betwen two vector
 *
 * @param[in] a  vector1
 * @param[in] b  vector2
 *
 * @return angle as radians
 */
CGLM_INLINE
float
glm_vec3_angle(vec3 a, vec3 b) {
  float norm, dot;

  /* maybe compiler generate approximation instruction (rcp) */
  norm = 1.0f / (glm_vec3_norm(a) * glm_vec3_norm(b));
  dot  = glm_vec3_dot(a, b) * norm;

  if (dot > 1.0f)
    return 0.0f;
  else if (dot < -1.0f)
    return CGLM_PI;

  return acosf(dot);
}

/*!
 * @brief rotate vec3 around axis by angle using Rodrigues' rotation formula
 *
 * @param[in, out] v     vector
 * @param[in]      axis  axis vector (must be unit vector)
 * @param[in]      angle angle by radians
 */
CGLM_INLINE
void
glm_vec3_rotate(vec3 v, float angle, vec3 axis) {
  vec3   v1, v2, k;
  float  c, s;

  c = cosf(angle);
  s = sinf(angle);

  glm_vec3_normalize_to(axis, k);

  /* Right Hand, Rodrigues' rotation formula:
        v = v*cos(t) + (kxv)sin(t) + k*(k.v)(1 - cos(t))
   */
  glm_vec3_scale(v, c, v1);

  glm_vec3_cross(k, v, v2);
  glm_vec3_scale(v2, s, v2);

  glm_vec3_add(v1, v2, v1);

  glm_vec3_scale(k, glm_vec3_dot(k, v) * (1.0f - c), v2);
  glm_vec3_add(v1, v2, v);
}

/*!
 * @brief apply rotation matrix to vector
 *
 *  matrix format should be (no perspective):
 *   a  b  c  x
 *   e  f  g  y
 *   i  j  k  z
 *   0  0  0  w
 *
 * @param[in]  m    affine matrix or rot matrix
 * @param[in]  v    vector
 * @param[out] dest rotated vector
 */
CGLM_INLINE
void
glm_vec3_rotate_m4(mat4 m, vec3 v, vec3 dest) {
  vec4 x, y, z, res;

  glm_vec4_normalize_to(m[0], x);
  glm_vec4_normalize_to(m[1], y);
  glm_vec4_normalize_to(m[2], z);

  glm_vec4_scale(x,   v[0], res);
  glm_vec4_muladds(y, v[1], res);
  glm_vec4_muladds(z, v[2], res);

  glm_vec3(res, dest);
}

/*!
 * @brief apply rotation matrix to vector
 *
 * @param[in]  m    affine matrix or rot matrix
 * @param[in]  v    vector
 * @param[out] dest rotated vector
 */
CGLM_INLINE
void
glm_vec3_rotate_m3(mat3 m, vec3 v, vec3 dest) {
  vec4 res, x, y, z;

  glm_vec4(m[0], 0.0f, x);
  glm_vec4(m[1], 0.0f, y);
  glm_vec4(m[2], 0.0f, z);

  glm_vec4_normalize(x);
  glm_vec4_normalize(y);
  glm_vec4_normalize(z);

  glm_vec4_scale(x,   v[0], res);
  glm_vec4_muladds(y, v[1], res);
  glm_vec4_muladds(z, v[2], res);

  glm_vec3(res, dest);
}

/*!
 * @brief project a vector onto b vector
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest projected vector
 */
CGLM_INLINE
void
glm_vec3_proj(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_scale(b,
                 glm_vec3_dot(a, b) / glm_vec3_norm2(b),
                 dest);
}

/**
 * @brief find center point of two vector
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest center point
 */
CGLM_INLINE
void
glm_vec3_center(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_add(a, b, dest);
  glm_vec3_scale(dest, 0.5f, dest);
}

/**
 * @brief squared distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return returns squared distance (distance * distance)
 */
CGLM_INLINE
float
glm_vec3_distance2(vec3 a, vec3 b) {
  return glm_pow2(a[0] - b[0])
       + glm_pow2(a[1] - b[1])
       + glm_pow2(a[2] - b[2]);
}

/**
 * @brief distance between two vectors
 *
 * @param[in] a vector1
 * @param[in] b vector2
 * @return returns distance
 */
CGLM_INLINE
float
glm_vec3_distance(vec3 a, vec3 b) {
  return sqrtf(glm_vec3_distance2(a, b));
}

/*!
 * @brief max values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec3_maxv(vec3 a, vec3 b, vec3 dest) {
  dest[0] = glm_max(a[0], b[0]);
  dest[1] = glm_max(a[1], b[1]);
  dest[2] = glm_max(a[2], b[2]);
}

/*!
 * @brief min values of vectors
 *
 * @param[in]  a    vector1
 * @param[in]  b    vector2
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec3_minv(vec3 a, vec3 b, vec3 dest) {
  dest[0] = glm_min(a[0], b[0]);
  dest[1] = glm_min(a[1], b[1]);
  dest[2] = glm_min(a[2], b[2]);
}

/*!
 * @brief possible orthogonal/perpendicular vector
 *
 * @param[in]  v    vector
 * @param[out] dest orthogonal/perpendicular vector
 */
CGLM_INLINE
void
glm_vec3_ortho(vec3 v, vec3 dest) {
  dest[0] = v[1] - v[2];
  dest[1] = v[2] - v[0];
  dest[2] = v[0] - v[1];
}

/*!
 * @brief clamp vector's individual members between min and max values
 *
 * @param[in, out]  v      vector
 * @param[in]       minVal minimum value
 * @param[in]       maxVal maximum value
 */
CGLM_INLINE
void
glm_vec3_clamp(vec3 v, float minVal, float maxVal) {
  v[0] = glm_clamp(v[0], minVal, maxVal);
  v[1] = glm_clamp(v[1], minVal, maxVal);
  v[2] = glm_clamp(v[2], minVal, maxVal);
}

/*!
 * @brief linear interpolation between two vectors
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount)
 * @param[out]  dest destination
 */
CGLM_INLINE
void
glm_vec3_lerp(vec3 from, vec3 to, float t, vec3 dest) {
  vec3 s, v;

  /* from + s * (to - from) */
  glm_vec3_broadcast(t, s);
  glm_vec3_sub(to, from, v);
  glm_vec3_mul(s, v, v);
  glm_vec3_add(from, v, dest);
}

/*!
 * @brief linear interpolation between two vectors (clamped)
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount) clamped between 0 and 1
 * @param[out]  dest destination
 */
CGLM_INLINE
void
glm_vec3_lerpc(vec3 from, vec3 to, float t, vec3 dest) {
  glm_vec3_lerp(from, to, glm_clamp_zo(t), dest);
}

/*!
 * @brief linear interpolation between two vectors
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount)
 * @param[out]  dest destination
 */
CGLM_INLINE
void
glm_vec3_mix(vec3 from, vec3 to, float t, vec3 dest) {
  glm_vec3_lerp(from, to, t, dest);
}

/*!
 * @brief linear interpolation between two vectors (clamped)
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount) clamped between 0 and 1
 * @param[out]  dest destination
 */
CGLM_INLINE
void
glm_vec3_mixc(vec3 from, vec3 to, float t, vec3 dest) {
  glm_vec3_lerpc(from, to, t, dest);
}

/*!
 * @brief threshold function (unidimensional)
 *
 * @param[in]   edge    threshold
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
CGLM_INLINE
void
glm_vec3_step_uni(float edge, vec3 x, vec3 dest) {
  dest[0] = glm_step(edge, x[0]);
  dest[1] = glm_step(edge, x[1]);
  dest[2] = glm_step(edge, x[2]);
}

/*!
 * @brief threshold function
 *
 * @param[in]   edge    threshold
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
CGLM_INLINE
void
glm_vec3_step(vec3 edge, vec3 x, vec3 dest) {
  dest[0] = glm_step(edge[0], x[0]);
  dest[1] = glm_step(edge[1], x[1]);
  dest[2] = glm_step(edge[2], x[2]);
}

/*!
 * @brief threshold function with a smooth transition (unidimensional)
 *
 * @param[in]   edge0   low threshold
 * @param[in]   edge1   high threshold
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
CGLM_INLINE
void
glm_vec3_smoothstep_uni(float edge0, float edge1, vec3 x, vec3 dest) {
  dest[0] = glm_smoothstep(edge0, edge1, x[0]);
  dest[1] = glm_smoothstep(edge0, edge1, x[1]);
  dest[2] = glm_smoothstep(edge0, edge1, x[2]);
}

/*!
 * @brief threshold function with a smooth transition
 *
 * @param[in]   edge0   low threshold
 * @param[in]   edge1   high threshold
 * @param[in]   x       value to test against threshold
 * @param[out]  dest    destination
 */
CGLM_INLINE
void
glm_vec3_smoothstep(vec3 edge0, vec3 edge1, vec3 x, vec3 dest) {
  dest[0] = glm_smoothstep(edge0[0], edge1[0], x[0]);
  dest[1] = glm_smoothstep(edge0[1], edge1[1], x[1]);
  dest[2] = glm_smoothstep(edge0[2], edge1[2], x[2]);
}

/*!
 * @brief smooth Hermite interpolation between two vectors
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount)
 * @param[out]  dest destination
 */
CGLM_INLINE
void
glm_vec3_smoothinterp(vec3 from, vec3 to, float t, vec3 dest) {
  vec3 s, v;
    
  /* from + s * (to - from) */
  glm_vec3_broadcast(glm_smooth(t), s);
  glm_vec3_sub(to, from, v);
  glm_vec3_mul(s, v, v);
  glm_vec3_add(from, v, dest);
}

/*!
 * @brief smooth Hermite interpolation between two vectors (clamped)
 *
 * formula:  from + s * (to - from)
 *
 * @param[in]   from from value
 * @param[in]   to   to value
 * @param[in]   t    interpolant (amount) clamped between 0 and 1
 * @param[out]  dest destination
 */
CGLM_INLINE
void
glm_vec3_smoothinterpc(vec3 from, vec3 to, float t, vec3 dest) {
  glm_vec3_smoothinterp(from, to, glm_clamp_zo(t), dest);
}

/*!
 * @brief swizzle vector components
 *
 * you can use existin masks e.g. GLM_XXX, GLM_ZYX
 *
 * @param[in]  v    source
 * @param[in]  mask mask
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_vec3_swizzle(vec3 v, int mask, vec3 dest) {
  vec3 t;

  t[0] = v[(mask & (3 << 0))];
  t[1] = v[(mask & (3 << 2)) >> 2];
  t[2] = v[(mask & (3 << 4)) >> 4];

  glm_vec3_copy(t, dest);
}

/*!
 * @brief vec3 cross product
 *
 * this is just convenient wrapper
 *
 * @param[in]  a source 1
 * @param[in]  b source 2
 * @param[out] d destination
 */
CGLM_INLINE
void
glm_cross(vec3 a, vec3 b, vec3 d) {
  glm_vec3_cross(a, b, d);
}

/*!
 * @brief vec3 dot product
 *
 * this is just convenient wrapper
 *
 * @param[in] a vector1
 * @param[in] b vector2
 *
 * @return dot product
 */
CGLM_INLINE
float
glm_dot(vec3 a, vec3 b) {
  return glm_vec3_dot(a, b);
}

/*!
 * @brief normalize vec3 and store result in same vec
 *
 * this is just convenient wrapper
 *
 * @param[in, out] v vector
 */
CGLM_INLINE
void
glm_normalize(vec3 v) {
  glm_vec3_normalize(v);
}

/*!
 * @brief normalize vec3 to dest
 *
 * this is just convenient wrapper
 *
 * @param[in]  v    source
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_normalize_to(vec3 v, vec3 dest) {
  glm_vec3_normalize_to(v, dest);
}

#endif /* cglm_vec3_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*!
 * Most of functions in this header are optimized manually with SIMD
 * if available. You dont need to call/incude SIMD headers manually
 */

/*
 Macros:
   GLM_MAT4_IDENTITY_INIT
   GLM_MAT4_ZERO_INIT
   GLM_MAT4_IDENTITY
   GLM_MAT4_ZERO

 Functions:
   CGLM_INLINE void  glm_mat4_ucopy(mat4 mat, mat4 dest);
   CGLM_INLINE void  glm_mat4_copy(mat4 mat, mat4 dest);
   CGLM_INLINE void  glm_mat4_identity(mat4 mat);
   CGLM_INLINE void  glm_mat4_identity_array(mat4 * restrict mat, size_t count);
   CGLM_INLINE void  glm_mat4_zero(mat4 mat);
   CGLM_INLINE void  glm_mat4_pick3(mat4 mat, mat3 dest);
   CGLM_INLINE void  glm_mat4_pick3t(mat4 mat, mat3 dest);
   CGLM_INLINE void  glm_mat4_ins3(mat3 mat, mat4 dest);
   CGLM_INLINE void  glm_mat4_mul(mat4 m1, mat4 m2, mat4 dest);
   CGLM_INLINE void  glm_mat4_mulN(mat4 *matrices[], int len, mat4 dest);
   CGLM_INLINE void  glm_mat4_mulv(mat4 m, vec4 v, vec4 dest);
   CGLM_INLINE void  glm_mat4_mulv3(mat4 m, vec3 v, vec3 dest);
   CGLM_INLINE float glm_mat4_trace(mat4 m);
   CGLM_INLINE float glm_mat4_trace3(mat4 m);
   CGLM_INLINE void  glm_mat4_quat(mat4 m, versor dest) ;
   CGLM_INLINE void  glm_mat4_transpose_to(mat4 m, mat4 dest);
   CGLM_INLINE void  glm_mat4_transpose(mat4 m);
   CGLM_INLINE void  glm_mat4_scale_p(mat4 m, float s);
   CGLM_INLINE void  glm_mat4_scale(mat4 m, float s);
   CGLM_INLINE float glm_mat4_det(mat4 mat);
   CGLM_INLINE void  glm_mat4_inv(mat4 mat, mat4 dest);
   CGLM_INLINE void  glm_mat4_inv_fast(mat4 mat, mat4 dest);
   CGLM_INLINE void  glm_mat4_swap_col(mat4 mat, int col1, int col2);
   CGLM_INLINE void  glm_mat4_swap_row(mat4 mat, int row1, int row2);
   CGLM_INLINE float glm_mat4_rmc(vec4 r, mat4 m, vec4 c);
 */

#ifndef cglm_mat_h
#define cglm_mat_h


#ifdef CGLM_SSE_FP
/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_mat_sse_h
#define cglm_mat_sse_h
#if defined( __SSE__ ) || defined( __SSE2__ )


#define glm_mat4_inv_precise_sse2(mat, dest) glm_mat4_inv_sse2(mat, dest)

CGLM_INLINE
void
glm_mat4_scale_sse2(mat4 m, float s) {
  __m128 x0;
  x0 = _mm_set1_ps(s);

  glmm_store(m[0], _mm_mul_ps(glmm_load(m[0]), x0));
  glmm_store(m[1], _mm_mul_ps(glmm_load(m[1]), x0));
  glmm_store(m[2], _mm_mul_ps(glmm_load(m[2]), x0));
  glmm_store(m[3], _mm_mul_ps(glmm_load(m[3]), x0));
}

CGLM_INLINE
void
glm_mat4_transp_sse2(mat4 m, mat4 dest) {
  __m128 r0, r1, r2, r3;

  r0 = glmm_load(m[0]);
  r1 = glmm_load(m[1]);
  r2 = glmm_load(m[2]);
  r3 = glmm_load(m[3]);

  _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

  glmm_store(dest[0], r0);
  glmm_store(dest[1], r1);
  glmm_store(dest[2], r2);
  glmm_store(dest[3], r3);
}

CGLM_INLINE
void
glm_mat4_mul_sse2(mat4 m1, mat4 m2, mat4 dest) {
  /* D = R * L (Column-Major) */

  __m128 l0, l1, l2, l3, r;

  l0 = glmm_load(m1[0]);
  l1 = glmm_load(m1[1]);
  l2 = glmm_load(m1[2]);
  l3 = glmm_load(m1[3]);

  r = glmm_load(m2[0]);
  glmm_store(dest[0],
             _mm_add_ps(_mm_add_ps(_mm_mul_ps(glmm_shuff1x(r, 0), l0),
                                   _mm_mul_ps(glmm_shuff1x(r, 1), l1)),
                        _mm_add_ps(_mm_mul_ps(glmm_shuff1x(r, 2), l2),
                                   _mm_mul_ps(glmm_shuff1x(r, 3), l3))));
  r = glmm_load(m2[1]);
  glmm_store(dest[1],
             _mm_add_ps(_mm_add_ps(_mm_mul_ps(glmm_shuff1x(r, 0), l0),
                                   _mm_mul_ps(glmm_shuff1x(r, 1), l1)),
                        _mm_add_ps(_mm_mul_ps(glmm_shuff1x(r, 2), l2),
                                   _mm_mul_ps(glmm_shuff1x(r, 3), l3))));
  r = glmm_load(m2[2]);
  glmm_store(dest[2],
             _mm_add_ps(_mm_add_ps(_mm_mul_ps(glmm_shuff1x(r, 0), l0),
                                   _mm_mul_ps(glmm_shuff1x(r, 1), l1)),
                        _mm_add_ps(_mm_mul_ps(glmm_shuff1x(r, 2), l2),
                                   _mm_mul_ps(glmm_shuff1x(r, 3), l3))));

  r = glmm_load(m2[3]);
  glmm_store(dest[3],
             _mm_add_ps(_mm_add_ps(_mm_mul_ps(glmm_shuff1x(r, 0), l0),
                                   _mm_mul_ps(glmm_shuff1x(r, 1), l1)),
                        _mm_add_ps(_mm_mul_ps(glmm_shuff1x(r, 2), l2),
                                   _mm_mul_ps(glmm_shuff1x(r, 3), l3))));
}

CGLM_INLINE
void
glm_mat4_mulv_sse2(mat4 m, vec4 v, vec4 dest) {
  __m128 x0, x1, x2;

  x0 = glmm_load(v);
  x1 = _mm_add_ps(_mm_mul_ps(glmm_load(m[0]), glmm_shuff1x(x0, 0)),
                  _mm_mul_ps(glmm_load(m[1]), glmm_shuff1x(x0, 1)));

  x2 = _mm_add_ps(_mm_mul_ps(glmm_load(m[2]), glmm_shuff1x(x0, 2)),
                  _mm_mul_ps(glmm_load(m[3]), glmm_shuff1x(x0, 3)));

  glmm_store(dest, _mm_add_ps(x1, x2));
}

CGLM_INLINE
float
glm_mat4_det_sse2(mat4 mat) {
  __m128 r0, r1, r2, r3, x0, x1, x2;

  /* 127 <- 0, [square] det(A) = det(At) */
  r0 = glmm_load(mat[0]); /* d c b a */
  r1 = glmm_load(mat[1]); /* h g f e */
  r2 = glmm_load(mat[2]); /* l k j i */
  r3 = glmm_load(mat[3]); /* p o n m */

  /*
   t[1] = j * p - n * l;
   t[2] = j * o - n * k;
   t[3] = i * p - m * l;
   t[4] = i * o - m * k;
   */
  x0 = _mm_sub_ps(_mm_mul_ps(glmm_shuff1(r2, 0, 0, 1, 1),
                             glmm_shuff1(r3, 2, 3, 2, 3)),
                  _mm_mul_ps(glmm_shuff1(r3, 0, 0, 1, 1),
                             glmm_shuff1(r2, 2, 3, 2, 3)));
  /*
   t[0] = k * p - o * l;
   t[0] = k * p - o * l;
   t[5] = i * n - m * j;
   t[5] = i * n - m * j;
   */
  x1 = _mm_sub_ps(_mm_mul_ps(glmm_shuff1(r2, 0, 0, 2, 2),
                             glmm_shuff1(r3, 1, 1, 3, 3)),
                  _mm_mul_ps(glmm_shuff1(r3, 0, 0, 2, 2),
                             glmm_shuff1(r2, 1, 1, 3, 3)));

  /*
     a * (f * t[0] - g * t[1] + h * t[2])
   - b * (e * t[0] - g * t[3] + h * t[4])
   + c * (e * t[1] - f * t[3] + h * t[5])
   - d * (e * t[2] - f * t[4] + g * t[5])
   */
  x2 = _mm_sub_ps(_mm_mul_ps(glmm_shuff1(r1, 0, 0, 0, 1),
                             _mm_shuffle_ps(x1, x0, _MM_SHUFFLE(1, 0, 0, 0))),
                  _mm_mul_ps(glmm_shuff1(r1, 1, 1, 2, 2),
                             glmm_shuff1(x0, 3, 2, 2, 0)));

  x2 = _mm_add_ps(x2,
                  _mm_mul_ps(glmm_shuff1(r1, 2, 3, 3, 3),
                             _mm_shuffle_ps(x0, x1, _MM_SHUFFLE(2, 2, 3, 1))));
  x2 = _mm_xor_ps(x2, _mm_set_ps(-0.f, 0.f, -0.f, 0.f));

  x0 = _mm_mul_ps(r0, x2);
  x0 = _mm_add_ps(x0, glmm_shuff1(x0, 0, 1, 2, 3));
  x0 = _mm_add_ps(x0, glmm_shuff1(x0, 1, 3, 3, 1));

  return _mm_cvtss_f32(x0);
}

CGLM_INLINE
void
glm_mat4_inv_fast_sse2(mat4 mat, mat4 dest) {
  __m128 r0, r1, r2, r3,
         v0, v1, v2, v3,
         t0, t1, t2, t3, t4, t5,
         x0, x1, x2, x3, x4, x5, x6, x7;

  /* 127 <- 0 */
  r0 = glmm_load(mat[0]); /* d c b a */
  r1 = glmm_load(mat[1]); /* h g f e */
  r2 = glmm_load(mat[2]); /* l k j i */
  r3 = glmm_load(mat[3]); /* p o n m */

  x0 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(3, 2, 3, 2));  /* p o l k */
  x1 = glmm_shuff1(x0, 1, 3, 3, 3);                      /* l p p p */
  x2 = glmm_shuff1(x0, 0, 2, 2, 2);                      /* k o o o */
  x0 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(3, 3, 3, 3));  /* h h l l */
  x3 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(2, 2, 2, 2));  /* g g k k */

  /* t1[0] = k * p - o * l;
     t1[0] = k * p - o * l;
     t2[0] = g * p - o * h;
     t3[0] = g * l - k * h; */
  t0 = _mm_sub_ps(_mm_mul_ps(x3, x1), _mm_mul_ps(x2, x0));

  x4 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(2, 1, 2, 1)); /* o n k j */
  x4 = glmm_shuff1(x4, 0, 2, 2, 2);                     /* j n n n */
  x5 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(1, 1, 1, 1)); /* f f j j */

  /* t1[1] = j * p - n * l;
     t1[1] = j * p - n * l;
     t2[1] = f * p - n * h;
     t3[1] = f * l - j * h; */
  t1 = _mm_sub_ps(_mm_mul_ps(x5, x1), _mm_mul_ps(x4, x0));

  /* t1[2] = j * o - n * k
     t1[2] = j * o - n * k;
     t2[2] = f * o - n * g;
     t3[2] = f * k - j * g; */
  t2 = _mm_sub_ps(_mm_mul_ps(x5, x2), _mm_mul_ps(x4, x3));

  x6 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(0, 0, 0, 0)); /* e e i i */
  x7 = glmm_shuff2(r3, r2, 0, 0, 0, 0, 2, 0, 0, 0);     /* i m m m */

  /* t1[3] = i * p - m * l;
     t1[3] = i * p - m * l;
     t2[3] = e * p - m * h;
     t3[3] = e * l - i * h; */
  t3 = _mm_sub_ps(_mm_mul_ps(x6, x1), _mm_mul_ps(x7, x0));

  /* t1[4] = i * o - m * k;
     t1[4] = i * o - m * k;
     t2[4] = e * o - m * g;
     t3[4] = e * k - i * g; */
  t4 = _mm_sub_ps(_mm_mul_ps(x6, x2), _mm_mul_ps(x7, x3));

  /* t1[5] = i * n - m * j;
     t1[5] = i * n - m * j;
     t2[5] = e * n - m * f;
     t3[5] = e * j - i * f; */
  t5 = _mm_sub_ps(_mm_mul_ps(x6, x4), _mm_mul_ps(x7, x5));

  x0 = glmm_shuff2(r1, r0, 0, 0, 0, 0, 2, 2, 2, 0); /* a a a e */
  x1 = glmm_shuff2(r1, r0, 1, 1, 1, 1, 2, 2, 2, 0); /* b b b f */
  x2 = glmm_shuff2(r1, r0, 2, 2, 2, 2, 2, 2, 2, 0); /* c c c g */
  x3 = glmm_shuff2(r1, r0, 3, 3, 3, 3, 2, 2, 2, 0); /* d d d h */

  /*
   dest[0][0] =  f * t1[0] - g * t1[1] + h * t1[2];
   dest[0][1] =-(b * t1[0] - c * t1[1] + d * t1[2]);
   dest[0][2] =  b * t2[0] - c * t2[1] + d * t2[2];
   dest[0][3] =-(b * t3[0] - c * t3[1] + d * t3[2]); */
  v0 = _mm_add_ps(_mm_mul_ps(x3, t2),
                  _mm_sub_ps(_mm_mul_ps(x1, t0),
                             _mm_mul_ps(x2, t1)));
  v0 = _mm_xor_ps(v0, _mm_set_ps(-0.f, 0.f, -0.f, 0.f));

  /*
   dest[1][0] =-(e * t1[0] - g * t1[3] + h * t1[4]);
   dest[1][1] =  a * t1[0] - c * t1[3] + d * t1[4];
   dest[1][2] =-(a * t2[0] - c * t2[3] + d * t2[4]);
   dest[1][3] =  a * t3[0] - c * t3[3] + d * t3[4]; */
  v1 = _mm_add_ps(_mm_mul_ps(x3, t4),
                  _mm_sub_ps(_mm_mul_ps(x0, t0),
                             _mm_mul_ps(x2, t3)));
  v1 = _mm_xor_ps(v1, _mm_set_ps(0.f, -0.f, 0.f, -0.f));

  /*
   dest[2][0] =  e * t1[1] - f * t1[3] + h * t1[5];
   dest[2][1] =-(a * t1[1] - b * t1[3] + d * t1[5]);
   dest[2][2] =  a * t2[1] - b * t2[3] + d * t2[5];
   dest[2][3] =-(a * t3[1] - b * t3[3] + d * t3[5]);*/
  v2 = _mm_add_ps(_mm_mul_ps(x3, t5),
                  _mm_sub_ps(_mm_mul_ps(x0, t1),
                             _mm_mul_ps(x1, t3)));
  v2 = _mm_xor_ps(v2, _mm_set_ps(-0.f, 0.f, -0.f, 0.f));

  /*
   dest[3][0] =-(e * t1[2] - f * t1[4] + g * t1[5]);
   dest[3][1] =  a * t1[2] - b * t1[4] + c * t1[5];
   dest[3][2] =-(a * t2[2] - b * t2[4] + c * t2[5]);
   dest[3][3] =  a * t3[2] - b * t3[4] + c * t3[5]; */
  v3 = _mm_add_ps(_mm_mul_ps(x2, t5),
                  _mm_sub_ps(_mm_mul_ps(x0, t2),
                             _mm_mul_ps(x1, t4)));
  v3 = _mm_xor_ps(v3, _mm_set_ps(0.f, -0.f, 0.f, -0.f));

  /* determinant */
  x0 = _mm_shuffle_ps(v0, v1, _MM_SHUFFLE(0, 0, 0, 0));
  x1 = _mm_shuffle_ps(v2, v3, _MM_SHUFFLE(0, 0, 0, 0));
  x0 = _mm_shuffle_ps(x0, x1, _MM_SHUFFLE(2, 0, 2, 0));

  x0 = _mm_mul_ps(x0, r0);
  x0 = _mm_add_ps(x0, glmm_shuff1(x0, 0, 1, 2, 3));
  x0 = _mm_add_ps(x0, glmm_shuff1(x0, 1, 0, 0, 1));
  x0 = _mm_rcp_ps(x0);

  glmm_store(dest[0], _mm_mul_ps(v0, x0));
  glmm_store(dest[1], _mm_mul_ps(v1, x0));
  glmm_store(dest[2], _mm_mul_ps(v2, x0));
  glmm_store(dest[3], _mm_mul_ps(v3, x0));
}

CGLM_INLINE
void
glm_mat4_inv_sse2(mat4 mat, mat4 dest) {
  __m128 r0, r1, r2, r3,
         v0, v1, v2, v3,
         t0, t1, t2, t3, t4, t5,
         x0, x1, x2, x3, x4, x5, x6, x7;

  /* 127 <- 0 */
  r0 = glmm_load(mat[0]); /* d c b a */
  r1 = glmm_load(mat[1]); /* h g f e */
  r2 = glmm_load(mat[2]); /* l k j i */
  r3 = glmm_load(mat[3]); /* p o n m */

  x0 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(3, 2, 3, 2));  /* p o l k */
  x1 = glmm_shuff1(x0, 1, 3, 3, 3);                      /* l p p p */
  x2 = glmm_shuff1(x0, 0, 2, 2, 2);                      /* k o o o */
  x0 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(3, 3, 3, 3));  /* h h l l */
  x3 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(2, 2, 2, 2));  /* g g k k */

  /* t1[0] = k * p - o * l;
     t1[0] = k * p - o * l;
     t2[0] = g * p - o * h;
     t3[0] = g * l - k * h; */
  t0 = _mm_sub_ps(_mm_mul_ps(x3, x1), _mm_mul_ps(x2, x0));

  x4 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(2, 1, 2, 1)); /* o n k j */
  x4 = glmm_shuff1(x4, 0, 2, 2, 2);                     /* j n n n */
  x5 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(1, 1, 1, 1)); /* f f j j */

  /* t1[1] = j * p - n * l;
     t1[1] = j * p - n * l;
     t2[1] = f * p - n * h;
     t3[1] = f * l - j * h; */
  t1 = _mm_sub_ps(_mm_mul_ps(x5, x1), _mm_mul_ps(x4, x0));

  /* t1[2] = j * o - n * k
     t1[2] = j * o - n * k;
     t2[2] = f * o - n * g;
     t3[2] = f * k - j * g; */
  t2 = _mm_sub_ps(_mm_mul_ps(x5, x2), _mm_mul_ps(x4, x3));

  x6 = _mm_shuffle_ps(r2, r1, _MM_SHUFFLE(0, 0, 0, 0)); /* e e i i */
  x7 = glmm_shuff2(r3, r2, 0, 0, 0, 0, 2, 0, 0, 0);     /* i m m m */

  /* t1[3] = i * p - m * l;
     t1[3] = i * p - m * l;
     t2[3] = e * p - m * h;
     t3[3] = e * l - i * h; */
  t3 = _mm_sub_ps(_mm_mul_ps(x6, x1), _mm_mul_ps(x7, x0));

  /* t1[4] = i * o - m * k;
     t1[4] = i * o - m * k;
     t2[4] = e * o - m * g;
     t3[4] = e * k - i * g; */
  t4 = _mm_sub_ps(_mm_mul_ps(x6, x2), _mm_mul_ps(x7, x3));

  /* t1[5] = i * n - m * j;
     t1[5] = i * n - m * j;
     t2[5] = e * n - m * f;
     t3[5] = e * j - i * f; */
  t5 = _mm_sub_ps(_mm_mul_ps(x6, x4), _mm_mul_ps(x7, x5));

  x0 = glmm_shuff2(r1, r0, 0, 0, 0, 0, 2, 2, 2, 0); /* a a a e */
  x1 = glmm_shuff2(r1, r0, 1, 1, 1, 1, 2, 2, 2, 0); /* b b b f */
  x2 = glmm_shuff2(r1, r0, 2, 2, 2, 2, 2, 2, 2, 0); /* c c c g */
  x3 = glmm_shuff2(r1, r0, 3, 3, 3, 3, 2, 2, 2, 0); /* d d d h */

  /*
   dest[0][0] =  f * t1[0] - g * t1[1] + h * t1[2];
   dest[0][1] =-(b * t1[0] - c * t1[1] + d * t1[2]);
   dest[0][2] =  b * t2[0] - c * t2[1] + d * t2[2];
   dest[0][3] =-(b * t3[0] - c * t3[1] + d * t3[2]); */
  v0 = _mm_add_ps(_mm_mul_ps(x3, t2),
                  _mm_sub_ps(_mm_mul_ps(x1, t0),
                             _mm_mul_ps(x2, t1)));
  v0 = _mm_xor_ps(v0, _mm_set_ps(-0.f, 0.f, -0.f, 0.f));

  /*
   dest[1][0] =-(e * t1[0] - g * t1[3] + h * t1[4]);
   dest[1][1] =  a * t1[0] - c * t1[3] + d * t1[4];
   dest[1][2] =-(a * t2[0] - c * t2[3] + d * t2[4]);
   dest[1][3] =  a * t3[0] - c * t3[3] + d * t3[4]; */
  v1 = _mm_add_ps(_mm_mul_ps(x3, t4),
                  _mm_sub_ps(_mm_mul_ps(x0, t0),
                             _mm_mul_ps(x2, t3)));
  v1 = _mm_xor_ps(v1, _mm_set_ps(0.f, -0.f, 0.f, -0.f));

  /*
   dest[2][0] =  e * t1[1] - f * t1[3] + h * t1[5];
   dest[2][1] =-(a * t1[1] - b * t1[3] + d * t1[5]);
   dest[2][2] =  a * t2[1] - b * t2[3] + d * t2[5];
   dest[2][3] =-(a * t3[1] - b * t3[3] + d * t3[5]);*/
  v2 = _mm_add_ps(_mm_mul_ps(x3, t5),
                  _mm_sub_ps(_mm_mul_ps(x0, t1),
                             _mm_mul_ps(x1, t3)));
  v2 = _mm_xor_ps(v2, _mm_set_ps(-0.f, 0.f, -0.f, 0.f));

  /*
   dest[3][0] =-(e * t1[2] - f * t1[4] + g * t1[5]);
   dest[3][1] =  a * t1[2] - b * t1[4] + c * t1[5];
   dest[3][2] =-(a * t2[2] - b * t2[4] + c * t2[5]);
   dest[3][3] =  a * t3[2] - b * t3[4] + c * t3[5]; */
  v3 = _mm_add_ps(_mm_mul_ps(x2, t5),
                  _mm_sub_ps(_mm_mul_ps(x0, t2),
                             _mm_mul_ps(x1, t4)));
  v3 = _mm_xor_ps(v3, _mm_set_ps(0.f, -0.f, 0.f, -0.f));

  /* determinant */
  x0 = _mm_shuffle_ps(v0, v1, _MM_SHUFFLE(0, 0, 0, 0));
  x1 = _mm_shuffle_ps(v2, v3, _MM_SHUFFLE(0, 0, 0, 0));
  x0 = _mm_shuffle_ps(x0, x1, _MM_SHUFFLE(2, 0, 2, 0));

  x0 = _mm_mul_ps(x0, r0);
  x0 = _mm_add_ps(x0, glmm_shuff1(x0, 0, 1, 2, 3));
  x0 = _mm_add_ps(x0, glmm_shuff1(x0, 1, 0, 0, 1));
  x0 = _mm_div_ps(_mm_set1_ps(1.0f), x0);

  glmm_store(dest[0], _mm_mul_ps(v0, x0));
  glmm_store(dest[1], _mm_mul_ps(v1, x0));
  glmm_store(dest[2], _mm_mul_ps(v2, x0));
  glmm_store(dest[3], _mm_mul_ps(v3, x0));
}

#endif
#endif /* cglm_mat_sse_h */

#endif

#ifdef CGLM_AVX_FP
/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_mat_simd_avx_h
#define cglm_mat_simd_avx_h
#ifdef __AVX__


#include <immintrin.h>

CGLM_INLINE
void
glm_mat4_mul_avx(mat4 m1, mat4 m2, mat4 dest) {
  /* D = R * L (Column-Major) */

  __m256 y0, y1, y2, y3, y4, y5, y6, y7, y8, y9;

  y0 = glmm_load256(m2[0]); /* h g f e d c b a */
  y1 = glmm_load256(m2[2]); /* p o n m l k j i */

  y2 = glmm_load256(m1[0]); /* h g f e d c b a */
  y3 = glmm_load256(m1[2]); /* p o n m l k j i */

  /* 0x03: 0b00000011 */
  y4 = _mm256_permute2f128_ps(y2, y2, 0x03); /* d c b a h g f e */
  y5 = _mm256_permute2f128_ps(y3, y3, 0x03); /* l k j i p o n m */

  /* f f f f a a a a */
  /* h h h h c c c c */
  /* e e e e b b b b */
  /* g g g g d d d d */
  y6 = _mm256_permutevar_ps(y0, _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0));
  y7 = _mm256_permutevar_ps(y0, _mm256_set_epi32(3, 3, 3, 3, 2, 2, 2, 2));
  y8 = _mm256_permutevar_ps(y0, _mm256_set_epi32(0, 0, 0, 0, 1, 1, 1, 1));
  y9 = _mm256_permutevar_ps(y0, _mm256_set_epi32(2, 2, 2, 2, 3, 3, 3, 3));

  glmm_store256(dest[0],
                _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(y2, y6),
                                            _mm256_mul_ps(y3, y7)),
                              _mm256_add_ps(_mm256_mul_ps(y4, y8),
                                            _mm256_mul_ps(y5, y9))));

  /* n n n n i i i i */
  /* p p p p k k k k */
  /* m m m m j j j j */
  /* o o o o l l l l */
  y6 = _mm256_permutevar_ps(y1, _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0));
  y7 = _mm256_permutevar_ps(y1, _mm256_set_epi32(3, 3, 3, 3, 2, 2, 2, 2));
  y8 = _mm256_permutevar_ps(y1, _mm256_set_epi32(0, 0, 0, 0, 1, 1, 1, 1));
  y9 = _mm256_permutevar_ps(y1, _mm256_set_epi32(2, 2, 2, 2, 3, 3, 3, 3));

  glmm_store256(dest[2],
                _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(y2, y6),
                                            _mm256_mul_ps(y3, y7)),
                              _mm256_add_ps(_mm256_mul_ps(y4, y8),
                                            _mm256_mul_ps(y5, y9))));
}

#endif
#endif /* cglm_mat_simd_avx_h */

#endif

#ifdef CGLM_NEON_FP
/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_mat4_neon_h
#define cglm_mat4_neon_h
#if defined(__ARM_NEON_FP)


CGLM_INLINE
void
glm_mat4_scale_neon(mat4 m, float s) {
  float32x4_t v0;
  
  v0 = vdupq_n_f32(s);

  vst1q_f32(m[0], vmulq_f32(vld1q_f32(m[0]), v0));
  vst1q_f32(m[1], vmulq_f32(vld1q_f32(m[1]), v0));
  vst1q_f32(m[2], vmulq_f32(vld1q_f32(m[2]), v0));
  vst1q_f32(m[3], vmulq_f32(vld1q_f32(m[3]), v0));
}

CGLM_INLINE
void
glm_mat4_transp_neon(mat4 m, mat4 dest) {
  float32x4x4_t vmat;
  
  vmat = vld4q_f32(m[0]);

  vst1q_f32(dest[0], vmat.val[0]);
  vst1q_f32(dest[1], vmat.val[1]);
  vst1q_f32(dest[2], vmat.val[2]);
  vst1q_f32(dest[3], vmat.val[3]);
}

CGLM_INLINE
void
glm_mat4_mul_neon(mat4 m1, mat4 m2, mat4 dest) {
  /* D = R * L (Column-Major) */
  float32x4_t l0, l1, l2, l3, r, d0, d1, d2, d3;

  l0 = vld1q_f32(m2[0]);
  l1 = vld1q_f32(m2[1]);
  l2 = vld1q_f32(m2[2]);
  l3 = vld1q_f32(m2[3]);

  r  = vld1q_f32(m1[0]);
  d0 = vmulq_lane_f32(r, vget_low_f32(l0), 0);
  d1 = vmulq_lane_f32(r, vget_low_f32(l1), 0);
  d2 = vmulq_lane_f32(r, vget_low_f32(l2), 0);
  d3 = vmulq_lane_f32(r, vget_low_f32(l3), 0);

  r  = vld1q_f32(m1[1]);
  d0 = vmlaq_lane_f32(d0, r, vget_low_f32(l0), 1);
  d1 = vmlaq_lane_f32(d1, r, vget_low_f32(l1), 1);
  d2 = vmlaq_lane_f32(d2, r, vget_low_f32(l2), 1);
  d3 = vmlaq_lane_f32(d3, r, vget_low_f32(l3), 1);

  r  = vld1q_f32(m1[2]);
  d0 = vmlaq_lane_f32(d0, r, vget_high_f32(l0), 0);
  d1 = vmlaq_lane_f32(d1, r, vget_high_f32(l1), 0);
  d2 = vmlaq_lane_f32(d2, r, vget_high_f32(l2), 0);
  d3 = vmlaq_lane_f32(d3, r, vget_high_f32(l3), 0);

  r  = vld1q_f32(m1[3]);
  d0 = vmlaq_lane_f32(d0, r, vget_high_f32(l0), 1);
  d1 = vmlaq_lane_f32(d1, r, vget_high_f32(l1), 1);
  d2 = vmlaq_lane_f32(d2, r, vget_high_f32(l2), 1);
  d3 = vmlaq_lane_f32(d3, r, vget_high_f32(l3), 1);

  vst1q_f32(dest[0], d0);
  vst1q_f32(dest[1], d1);
  vst1q_f32(dest[2], d2);
  vst1q_f32(dest[3], d3);
}

CGLM_INLINE
void
glm_mat4_mulv_neon(mat4 m, vec4 v, vec4 dest) {
  float32x4_t l0, l1, l2, l3;
  float32x2_t vlo, vhi;
  
  l0  = vld1q_f32(m[0]);
  l1  = vld1q_f32(m[1]);
  l2  = vld1q_f32(m[2]);
  l3  = vld1q_f32(m[3]);

  vlo = vld1_f32(&v[0]);
  vhi = vld1_f32(&v[2]);

  l0  = vmulq_lane_f32(l0, vlo, 0);
  l0  = vmlaq_lane_f32(l0, l1, vlo, 1);
  l0  = vmlaq_lane_f32(l0, l2, vhi, 0);
  l0  = vmlaq_lane_f32(l0, l3, vhi, 1);

  vst1q_f32(dest, l0);
}

#endif
#endif /* cglm_mat4_neon_h */

#endif

#ifdef DEBUG
# include <assert.h>
#endif

#define GLM_MAT4_IDENTITY_INIT  {{1.0f, 0.0f, 0.0f, 0.0f},                    \
                                 {0.0f, 1.0f, 0.0f, 0.0f},                    \
                                 {0.0f, 0.0f, 1.0f, 0.0f},                    \
                                 {0.0f, 0.0f, 0.0f, 1.0f}}

#define GLM_MAT4_ZERO_INIT      {{0.0f, 0.0f, 0.0f, 0.0f},                    \
                                 {0.0f, 0.0f, 0.0f, 0.0f},                    \
                                 {0.0f, 0.0f, 0.0f, 0.0f},                    \
                                 {0.0f, 0.0f, 0.0f, 0.0f}}

/* for C only */
#define GLM_MAT4_IDENTITY ((mat4)GLM_MAT4_IDENTITY_INIT)
#define GLM_MAT4_ZERO     ((mat4)GLM_MAT4_ZERO_INIT)

/* DEPRECATED! use _copy, _ucopy versions */
#define glm_mat4_udup(mat, dest) glm_mat4_ucopy(mat, dest)
#define glm_mat4_dup(mat, dest)  glm_mat4_copy(mat, dest)

/* DEPRECATED! default is precise now. */
#define glm_mat4_inv_precise(mat, dest) glm_mat4_inv(mat, dest)

/*!
 * @brief copy all members of [mat] to [dest]
 *
 * matrix may not be aligned, u stands for unaligned, this may be useful when
 * copying a matrix from external source e.g. asset importer...
 *
 * @param[in]  mat  source
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_mat4_ucopy(mat4 mat, mat4 dest) {
  dest[0][0] = mat[0][0];  dest[1][0] = mat[1][0];
  dest[0][1] = mat[0][1];  dest[1][1] = mat[1][1];
  dest[0][2] = mat[0][2];  dest[1][2] = mat[1][2];
  dest[0][3] = mat[0][3];  dest[1][3] = mat[1][3];

  dest[2][0] = mat[2][0];  dest[3][0] = mat[3][0];
  dest[2][1] = mat[2][1];  dest[3][1] = mat[3][1];
  dest[2][2] = mat[2][2];  dest[3][2] = mat[3][2];
  dest[2][3] = mat[2][3];  dest[3][3] = mat[3][3];
}

/*!
 * @brief copy all members of [mat] to [dest]
 *
 * @param[in]  mat  source
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_mat4_copy(mat4 mat, mat4 dest) {
#ifdef __AVX__
  glmm_store256(dest[0], glmm_load256(mat[0]));
  glmm_store256(dest[2], glmm_load256(mat[2]));
#elif defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(dest[0], glmm_load(mat[0]));
  glmm_store(dest[1], glmm_load(mat[1]));
  glmm_store(dest[2], glmm_load(mat[2]));
  glmm_store(dest[3], glmm_load(mat[3]));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(dest[0], vld1q_f32(mat[0]));
  vst1q_f32(dest[1], vld1q_f32(mat[1]));
  vst1q_f32(dest[2], vld1q_f32(mat[2]));
  vst1q_f32(dest[3], vld1q_f32(mat[3]));
#else
  glm_mat4_ucopy(mat, dest);
#endif
}

/*!
 * @brief make given matrix identity. It is identical with below, 
 *        but it is more easy to do that with this func especially for members
 *        e.g. glm_mat4_identity(aStruct->aMatrix);
 *
 * @code
 * glm_mat4_copy(GLM_MAT4_IDENTITY, mat); // C only
 *
 * // or
 * mat4 mat = GLM_MAT4_IDENTITY_INIT;
 * @endcode
 *
 * @param[in, out]  mat  destination
 */
CGLM_INLINE
void
glm_mat4_identity(mat4 mat) {
  CGLM_ALIGN_MAT mat4 t = GLM_MAT4_IDENTITY_INIT;
  glm_mat4_copy(t, mat);
}

/*!
 * @brief make given matrix array's each element identity matrix
 *
 * @param[in, out]  mat   matrix array (must be aligned (16/32)
 *                        if alignment is not disabled)
 *
 * @param[in]       count count of matrices
 */
CGLM_INLINE
void
glm_mat4_identity_array(mat4 * __restrict mat, size_t count) {
  CGLM_ALIGN_MAT mat4 t = GLM_MAT4_IDENTITY_INIT;
  size_t i;

  for (i = 0; i < count; i++) {
    glm_mat4_copy(t, mat[i]);
  }
}

/*!
 * @brief make given matrix zero.
 *
 * @param[in, out]  mat  matrix
 */
CGLM_INLINE
void
glm_mat4_zero(mat4 mat) {
  CGLM_ALIGN_MAT mat4 t = GLM_MAT4_ZERO_INIT;
  glm_mat4_copy(t, mat);
}

/*!
 * @brief copy upper-left of mat4 to mat3
 *
 * @param[in]  mat  source
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_mat4_pick3(mat4 mat, mat3 dest) {
  dest[0][0] = mat[0][0];
  dest[0][1] = mat[0][1];
  dest[0][2] = mat[0][2];

  dest[1][0] = mat[1][0];
  dest[1][1] = mat[1][1];
  dest[1][2] = mat[1][2];

  dest[2][0] = mat[2][0];
  dest[2][1] = mat[2][1];
  dest[2][2] = mat[2][2];
}

/*!
 * @brief copy upper-left of mat4 to mat3 (transposed)
 *
 * the postfix t stands for transpose
 *
 * @param[in]  mat  source
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_mat4_pick3t(mat4 mat, mat3 dest) {
  dest[0][0] = mat[0][0];
  dest[0][1] = mat[1][0];
  dest[0][2] = mat[2][0];

  dest[1][0] = mat[0][1];
  dest[1][1] = mat[1][1];
  dest[1][2] = mat[2][1];

  dest[2][0] = mat[0][2];
  dest[2][1] = mat[1][2];
  dest[2][2] = mat[2][2];
}

/*!
 * @brief copy mat3 to mat4's upper-left
 *
 * @param[in]  mat  source
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_mat4_ins3(mat3 mat, mat4 dest) {
  dest[0][0] = mat[0][0];
  dest[0][1] = mat[0][1];
  dest[0][2] = mat[0][2];

  dest[1][0] = mat[1][0];
  dest[1][1] = mat[1][1];
  dest[1][2] = mat[1][2];

  dest[2][0] = mat[2][0];
  dest[2][1] = mat[2][1];
  dest[2][2] = mat[2][2];
}

/*!
 * @brief multiply m1 and m2 to dest
 *
 * m1, m2 and dest matrices can be same matrix, it is possible to write this:
 *
 * @code
 * mat4 m = GLM_MAT4_IDENTITY_INIT;
 * glm_mat4_mul(m, m, m);
 * @endcode
 *
 * @param[in]  m1   left matrix
 * @param[in]  m2   right matrix
 * @param[out] dest destination matrix
 */
CGLM_INLINE
void
glm_mat4_mul(mat4 m1, mat4 m2, mat4 dest) {
#ifdef __AVX__
  glm_mat4_mul_avx(m1, m2, dest);
#elif defined( __SSE__ ) || defined( __SSE2__ )
  glm_mat4_mul_sse2(m1, m2, dest);
#elif defined(CGLM_NEON_FP)
  glm_mat4_mul_neon(m1, m2, dest);
#else
  float a00 = m1[0][0], a01 = m1[0][1], a02 = m1[0][2], a03 = m1[0][3],
        a10 = m1[1][0], a11 = m1[1][1], a12 = m1[1][2], a13 = m1[1][3],
        a20 = m1[2][0], a21 = m1[2][1], a22 = m1[2][2], a23 = m1[2][3],
        a30 = m1[3][0], a31 = m1[3][1], a32 = m1[3][2], a33 = m1[3][3],

        b00 = m2[0][0], b01 = m2[0][1], b02 = m2[0][2], b03 = m2[0][3],
        b10 = m2[1][0], b11 = m2[1][1], b12 = m2[1][2], b13 = m2[1][3],
        b20 = m2[2][0], b21 = m2[2][1], b22 = m2[2][2], b23 = m2[2][3],
        b30 = m2[3][0], b31 = m2[3][1], b32 = m2[3][2], b33 = m2[3][3];

  dest[0][0] = a00 * b00 + a10 * b01 + a20 * b02 + a30 * b03;
  dest[0][1] = a01 * b00 + a11 * b01 + a21 * b02 + a31 * b03;
  dest[0][2] = a02 * b00 + a12 * b01 + a22 * b02 + a32 * b03;
  dest[0][3] = a03 * b00 + a13 * b01 + a23 * b02 + a33 * b03;
  dest[1][0] = a00 * b10 + a10 * b11 + a20 * b12 + a30 * b13;
  dest[1][1] = a01 * b10 + a11 * b11 + a21 * b12 + a31 * b13;
  dest[1][2] = a02 * b10 + a12 * b11 + a22 * b12 + a32 * b13;
  dest[1][3] = a03 * b10 + a13 * b11 + a23 * b12 + a33 * b13;
  dest[2][0] = a00 * b20 + a10 * b21 + a20 * b22 + a30 * b23;
  dest[2][1] = a01 * b20 + a11 * b21 + a21 * b22 + a31 * b23;
  dest[2][2] = a02 * b20 + a12 * b21 + a22 * b22 + a32 * b23;
  dest[2][3] = a03 * b20 + a13 * b21 + a23 * b22 + a33 * b23;
  dest[3][0] = a00 * b30 + a10 * b31 + a20 * b32 + a30 * b33;
  dest[3][1] = a01 * b30 + a11 * b31 + a21 * b32 + a31 * b33;
  dest[3][2] = a02 * b30 + a12 * b31 + a22 * b32 + a32 * b33;
  dest[3][3] = a03 * b30 + a13 * b31 + a23 * b32 + a33 * b33;
#endif
}

/*!
 * @brief mupliply N mat4 matrices and store result in dest
 *
 * this function lets you multiply multiple (more than two or more...) matrices
 * <br><br>multiplication will be done in loop, this may reduce instructions
 * size but if <b>len</b> is too small then compiler may unroll whole loop,
 * usage:
 * @code
 * mat m1, m2, m3, m4, res;
 *
 * glm_mat4_mulN((mat4 *[]){&m1, &m2, &m3, &m4}, 4, res);
 * @endcode
 *
 * @warning matrices parameter is pointer array not mat4 array!
 *
 * @param[in]  matrices mat4 * array
 * @param[in]  len      matrices count
 * @param[out] dest     result
 */
CGLM_INLINE
void
glm_mat4_mulN(mat4 * __restrict matrices[], uint32_t len, mat4 dest) {
  uint32_t i;

#ifdef DEBUG
  assert(len > 1 && "there must be least 2 matrices to go!");
#endif

  glm_mat4_mul(*matrices[0], *matrices[1], dest);

  for (i = 2; i < len; i++)
    glm_mat4_mul(dest, *matrices[i], dest);
}

/*!
 * @brief multiply mat4 with vec4 (column vector) and store in dest vector
 *
 * @param[in]  m    mat4 (left)
 * @param[in]  v    vec4 (right, column vector)
 * @param[out] dest vec4 (result, column vector)
 */
CGLM_INLINE
void
glm_mat4_mulv(mat4 m, vec4 v, vec4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glm_mat4_mulv_sse2(m, v, dest);
#elif defined(CGLM_NEON_FP)
  glm_mat4_mulv_neon(m, v, dest);
#else
  vec4 res;
  res[0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0] * v[3];
  res[1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1] * v[3];
  res[2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2] * v[3];
  res[3] = m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3] * v[3];
  glm_vec4_copy(res, dest);
#endif
}

/*!
 * @brief trace of matrix
 *
 * sum of the elements on the main diagonal from upper left to the lower right
 *
 * @param[in]  m matrix
 */
CGLM_INLINE
float
glm_mat4_trace(mat4 m) {
  return m[0][0] + m[1][1] + m[2][2] + m[3][3];
}

/*!
 * @brief trace of matrix (rotation part)
 *
 * sum of the elements on the main diagonal from upper left to the lower right
 *
 * @param[in]  m matrix
 */
CGLM_INLINE
float
glm_mat4_trace3(mat4 m) {
  return m[0][0] + m[1][1] + m[2][2];
}

/*!
 * @brief convert mat4's rotation part to quaternion
 *
 * @param[in]  m    affine matrix
 * @param[out] dest destination quaternion
 */
CGLM_INLINE
void
glm_mat4_quat(mat4 m, versor dest) {
  float trace, r, rinv;

  /* it seems using like m12 instead of m[1][2] causes extra instructions */

  trace = m[0][0] + m[1][1] + m[2][2];
  if (trace >= 0.0f) {
    r       = sqrtf(1.0f + trace);
    rinv    = 0.5f / r;

    dest[0] = rinv * (m[1][2] - m[2][1]);
    dest[1] = rinv * (m[2][0] - m[0][2]);
    dest[2] = rinv * (m[0][1] - m[1][0]);
    dest[3] = r    * 0.5f;
  } else if (m[0][0] >= m[1][1] && m[0][0] >= m[2][2]) {
    r       = sqrtf(1.0f - m[1][1] - m[2][2] + m[0][0]);
    rinv    = 0.5f / r;

    dest[0] = r    * 0.5f;
    dest[1] = rinv * (m[0][1] + m[1][0]);
    dest[2] = rinv * (m[0][2] + m[2][0]);
    dest[3] = rinv * (m[1][2] - m[2][1]);
  } else if (m[1][1] >= m[2][2]) {
    r       = sqrtf(1.0f - m[0][0] - m[2][2] + m[1][1]);
    rinv    = 0.5f / r;

    dest[0] = rinv * (m[0][1] + m[1][0]);
    dest[1] = r    * 0.5f;
    dest[2] = rinv * (m[1][2] + m[2][1]);
    dest[3] = rinv * (m[2][0] - m[0][2]);
  } else {
    r       = sqrtf(1.0f - m[0][0] - m[1][1] + m[2][2]);
    rinv    = 0.5f / r;

    dest[0] = rinv * (m[0][2] + m[2][0]);
    dest[1] = rinv * (m[1][2] + m[2][1]);
    dest[2] = r    * 0.5f;
    dest[3] = rinv * (m[0][1] - m[1][0]);
  }
}

/*!
 * @brief multiply vector with mat4
 *
 * actually the result is vec4, after multiplication the last component
 * is trimmed. if you need it don't use this func.
 *
 * @param[in]  m    mat4(affine transform)
 * @param[in]  v    vec3
 * @param[in]  last 4th item to make it vec4
 * @param[out] dest result vector (vec3)
 */
CGLM_INLINE
void
glm_mat4_mulv3(mat4 m, vec3 v, float last, vec3 dest) {
  vec4 res;
  glm_vec4(v, last, res);
  glm_mat4_mulv(m, res, res);
  glm_vec3(res, dest);
}

/*!
 * @brief transpose mat4 and store in dest
 *
 * source matrix will not be transposed unless dest is m
 *
 * @param[in]  m    matrix
 * @param[out] dest result
 */
CGLM_INLINE
void
glm_mat4_transpose_to(mat4 m, mat4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glm_mat4_transp_sse2(m, dest);
#elif defined(CGLM_NEON_FP)
  glm_mat4_transp_neon(m, dest);
#else
  dest[0][0] = m[0][0]; dest[1][0] = m[0][1];
  dest[0][1] = m[1][0]; dest[1][1] = m[1][1];
  dest[0][2] = m[2][0]; dest[1][2] = m[2][1];
  dest[0][3] = m[3][0]; dest[1][3] = m[3][1];
  dest[2][0] = m[0][2]; dest[3][0] = m[0][3];
  dest[2][1] = m[1][2]; dest[3][1] = m[1][3];
  dest[2][2] = m[2][2]; dest[3][2] = m[2][3];
  dest[2][3] = m[3][2]; dest[3][3] = m[3][3];
#endif
}

/*!
 * @brief tranpose mat4 and store result in same matrix
 *
 * @param[in, out] m source and dest
 */
CGLM_INLINE
void
glm_mat4_transpose(mat4 m) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glm_mat4_transp_sse2(m, m);
#elif defined(CGLM_NEON_FP)
  glm_mat4_transp_neon(m, m);
#else
  mat4 d;
  glm_mat4_transpose_to(m, d);
  glm_mat4_ucopy(d, m);
#endif
}

/*!
 * @brief scale (multiply with scalar) matrix without simd optimization
 *
 * multiply matrix with scalar
 *
 * @param[in, out] m matrix
 * @param[in]      s scalar
 */
CGLM_INLINE
void
glm_mat4_scale_p(mat4 m, float s) {
  m[0][0] *= s; m[0][1] *= s; m[0][2] *= s; m[0][3] *= s;
  m[1][0] *= s; m[1][1] *= s; m[1][2] *= s; m[1][3] *= s;
  m[2][0] *= s; m[2][1] *= s; m[2][2] *= s; m[2][3] *= s;
  m[3][0] *= s; m[3][1] *= s; m[3][2] *= s; m[3][3] *= s;
}

/*!
 * @brief scale (multiply with scalar) matrix
 *
 * multiply matrix with scalar
 *
 * @param[in, out] m matrix
 * @param[in]      s scalar
 */
CGLM_INLINE
void
glm_mat4_scale(mat4 m, float s) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glm_mat4_scale_sse2(m, s);
#elif defined(CGLM_NEON_FP)
  glm_mat4_scale_neon(m, s);
#else
  glm_mat4_scale_p(m, s);
#endif
}

/*!
 * @brief mat4 determinant
 *
 * @param[in] mat matrix
 *
 * @return determinant
 */
CGLM_INLINE
float
glm_mat4_det(mat4 mat) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  return glm_mat4_det_sse2(mat);
#else
  /* [square] det(A) = det(At) */
  float t[6];
  float a = mat[0][0], b = mat[0][1], c = mat[0][2], d = mat[0][3],
        e = mat[1][0], f = mat[1][1], g = mat[1][2], h = mat[1][3],
        i = mat[2][0], j = mat[2][1], k = mat[2][2], l = mat[2][3],
        m = mat[3][0], n = mat[3][1], o = mat[3][2], p = mat[3][3];

  t[0] = k * p - o * l;
  t[1] = j * p - n * l;
  t[2] = j * o - n * k;
  t[3] = i * p - m * l;
  t[4] = i * o - m * k;
  t[5] = i * n - m * j;

  return a * (f * t[0] - g * t[1] + h * t[2])
       - b * (e * t[0] - g * t[3] + h * t[4])
       + c * (e * t[1] - f * t[3] + h * t[5])
       - d * (e * t[2] - f * t[4] + g * t[5]);
#endif
}

/*!
 * @brief inverse mat4 and store in dest
 *
 * @param[in]  mat  matrix
 * @param[out] dest inverse matrix
 */
CGLM_INLINE
void
glm_mat4_inv(mat4 mat, mat4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glm_mat4_inv_sse2(mat, dest);
#else
  float t[6];
  float det;
  float a = mat[0][0], b = mat[0][1], c = mat[0][2], d = mat[0][3],
        e = mat[1][0], f = mat[1][1], g = mat[1][2], h = mat[1][3],
        i = mat[2][0], j = mat[2][1], k = mat[2][2], l = mat[2][3],
        m = mat[3][0], n = mat[3][1], o = mat[3][2], p = mat[3][3];

  t[0] = k * p - o * l; t[1] = j * p - n * l; t[2] = j * o - n * k;
  t[3] = i * p - m * l; t[4] = i * o - m * k; t[5] = i * n - m * j;

  dest[0][0] =  f * t[0] - g * t[1] + h * t[2];
  dest[1][0] =-(e * t[0] - g * t[3] + h * t[4]);
  dest[2][0] =  e * t[1] - f * t[3] + h * t[5];
  dest[3][0] =-(e * t[2] - f * t[4] + g * t[5]);

  dest[0][1] =-(b * t[0] - c * t[1] + d * t[2]);
  dest[1][1] =  a * t[0] - c * t[3] + d * t[4];
  dest[2][1] =-(a * t[1] - b * t[3] + d * t[5]);
  dest[3][1] =  a * t[2] - b * t[4] + c * t[5];

  t[0] = g * p - o * h; t[1] = f * p - n * h; t[2] = f * o - n * g;
  t[3] = e * p - m * h; t[4] = e * o - m * g; t[5] = e * n - m * f;

  dest[0][2] =  b * t[0] - c * t[1] + d * t[2];
  dest[1][2] =-(a * t[0] - c * t[3] + d * t[4]);
  dest[2][2] =  a * t[1] - b * t[3] + d * t[5];
  dest[3][2] =-(a * t[2] - b * t[4] + c * t[5]);

  t[0] = g * l - k * h; t[1] = f * l - j * h; t[2] = f * k - j * g;
  t[3] = e * l - i * h; t[4] = e * k - i * g; t[5] = e * j - i * f;

  dest[0][3] =-(b * t[0] - c * t[1] + d * t[2]);
  dest[1][3] =  a * t[0] - c * t[3] + d * t[4];
  dest[2][3] =-(a * t[1] - b * t[3] + d * t[5]);
  dest[3][3] =  a * t[2] - b * t[4] + c * t[5];

  det = 1.0f / (a * dest[0][0] + b * dest[1][0]
              + c * dest[2][0] + d * dest[3][0]);

  glm_mat4_scale_p(dest, det);
#endif
}

/*!
 * @brief inverse mat4 and store in dest
 *
 * this func uses reciprocal approximation without extra corrections
 * e.g Newton-Raphson. this should work faster than normal,
 * to get more precise use glm_mat4_inv version.
 *
 * NOTE: You will lose precision, glm_mat4_inv is more accurate
 *
 * @param[in]  mat  matrix
 * @param[out] dest inverse matrix
 */
CGLM_INLINE
void
glm_mat4_inv_fast(mat4 mat, mat4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glm_mat4_inv_fast_sse2(mat, dest);
#else
  glm_mat4_inv(mat, dest);
#endif
}

/*!
 * @brief swap two matrix columns
 *
 * @param[in,out] mat  matrix
 * @param[in]     col1 col1
 * @param[in]     col2 col2
 */
CGLM_INLINE
void
glm_mat4_swap_col(mat4 mat, int col1, int col2) {
  CGLM_ALIGN(16) vec4 tmp;
  glm_vec4_copy(mat[col1], tmp);
  glm_vec4_copy(mat[col2], mat[col1]);
  glm_vec4_copy(tmp, mat[col2]);
}

/*!
 * @brief swap two matrix rows
 *
 * @param[in,out] mat  matrix
 * @param[in]     row1 row1
 * @param[in]     row2 row2
 */
CGLM_INLINE
void
glm_mat4_swap_row(mat4 mat, int row1, int row2) {
  CGLM_ALIGN(16) vec4 tmp;
  tmp[0] = mat[0][row1];
  tmp[1] = mat[1][row1];
  tmp[2] = mat[2][row1];
  tmp[3] = mat[3][row1];

  mat[0][row1] = mat[0][row2];
  mat[1][row1] = mat[1][row2];
  mat[2][row1] = mat[2][row2];
  mat[3][row1] = mat[3][row2];

  mat[0][row2] = tmp[0];
  mat[1][row2] = tmp[1];
  mat[2][row2] = tmp[2];
  mat[3][row2] = tmp[3];
}

/*!
 * @brief helper for  R (row vector) * M (matrix) * C (column vector)
 *
 * rmc stands for Row * Matrix * Column
 *
 * the result is scalar because R * M = Matrix1x4 (row vector),
 * then Matrix1x4 * Vec4 (column vector) = Matrix1x1 (Scalar)
 *
 * @param[in]  r   row vector or matrix1x4
 * @param[in]  m   matrix4x4
 * @param[in]  c   column vector or matrix4x1
 *
 * @return scalar value e.g. B(s)
 */
CGLM_INLINE
float
glm_mat4_rmc(vec4 r, mat4 m, vec4 c) {
  vec4 tmp;
  glm_mat4_mulv(m, c, tmp);
  return glm_vec4_dot(r, tmp);
}

#endif /* cglm_mat_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*
 Macros:
   GLM_MAT3_IDENTITY_INIT
   GLM_MAT3_ZERO_INIT
   GLM_MAT3_IDENTITY
   GLM_MAT3_ZERO
   glm_mat3_dup(mat, dest)

 Functions:
   CGLM_INLINE void  glm_mat3_copy(mat3 mat, mat3 dest);
   CGLM_INLINE void  glm_mat3_identity(mat3 mat);
   CGLM_INLINE void  glm_mat3_identity_array(mat3 * restrict mat, size_t count);
   CGLM_INLINE void  glm_mat3_zero(mat3 mat);
   CGLM_INLINE void  glm_mat3_mul(mat3 m1, mat3 m2, mat3 dest);
   CGLM_INLINE void  glm_mat3_transpose_to(mat3 m, mat3 dest);
   CGLM_INLINE void  glm_mat3_transpose(mat3 m);
   CGLM_INLINE void  glm_mat3_mulv(mat3 m, vec3 v, vec3 dest);
   CGLM_INLINE float glm_mat3_trace(mat3 m);
   CGLM_INLINE void  glm_mat3_quat(mat3 m, versor dest);
   CGLM_INLINE void  glm_mat3_scale(mat3 m, float s);
   CGLM_INLINE float glm_mat3_det(mat3 mat);
   CGLM_INLINE void  glm_mat3_inv(mat3 mat, mat3 dest);
   CGLM_INLINE void  glm_mat3_swap_col(mat3 mat, int col1, int col2);
   CGLM_INLINE void  glm_mat3_swap_row(mat3 mat, int row1, int row2);
   CGLM_INLINE float glm_mat3_rmc(vec3 r, mat3 m, vec3 c);
 */

#ifndef cglm_mat3_h
#define cglm_mat3_h


#ifdef CGLM_SSE_FP
/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_mat3_sse_h
#define cglm_mat3_sse_h
#if defined( __SSE__ ) || defined( __SSE2__ )


CGLM_INLINE
void
glm_mat3_mul_sse2(mat3 m1, mat3 m2, mat3 dest) {
  __m128 l0, l1, l2;
  __m128 r0, r1, r2;
  __m128 x0, x1, x2;

  l0 = _mm_loadu_ps(m1[0]);
  l1 = _mm_loadu_ps(&m1[1][1]);
  l2 = _mm_set1_ps(m1[2][2]);

  r0 = _mm_loadu_ps(m2[0]);
  r1 = _mm_loadu_ps(&m2[1][1]);
  r2 = _mm_set1_ps(m2[2][2]);

  x1 = glmm_shuff2(l0, l1, 1, 0, 3, 3, 0, 3, 2, 0);
  x2 = glmm_shuff2(l1, l2, 0, 0, 3, 2, 0, 2, 1, 0);

  x0 = _mm_add_ps(_mm_mul_ps(glmm_shuff1(l0, 0, 2, 1, 0),
                             glmm_shuff1(r0, 3, 0, 0, 0)),
                  _mm_mul_ps(x1, glmm_shuff2(r0, r1, 0, 0, 1, 1, 2, 0, 0, 0)));

  x0 = _mm_add_ps(x0,
                  _mm_mul_ps(x2, glmm_shuff2(r0, r1, 1, 1, 2, 2, 2, 0, 0, 0)));

  _mm_storeu_ps(dest[0], x0);

  x0 = _mm_add_ps(_mm_mul_ps(glmm_shuff1(l0, 1, 0, 2, 1),
                             _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 2, 3, 3))),
                  _mm_mul_ps(glmm_shuff1(x1, 1, 0, 2, 1),
                             glmm_shuff1(r1, 3, 3, 0, 0)));

  x0 = _mm_add_ps(x0,
                  _mm_mul_ps(glmm_shuff1(x2, 1, 0, 2, 1),
                             _mm_shuffle_ps(r1, r2, _MM_SHUFFLE(0, 0, 1, 1))));

  _mm_storeu_ps(&dest[1][1], x0);

  dest[2][2] = m1[0][2] * m2[2][0]
             + m1[1][2] * m2[2][1]
             + m1[2][2] * m2[2][2];
}

#endif
#endif /* cglm_mat3_sse_h */

#endif

#define GLM_MAT3_IDENTITY_INIT  {{1.0f, 0.0f, 0.0f},                          \
                                 {0.0f, 1.0f, 0.0f},                          \
                                 {0.0f, 0.0f, 1.0f}}
#define GLM_MAT3_ZERO_INIT      {{0.0f, 0.0f, 0.0f},                          \
                                 {0.0f, 0.0f, 0.0f},                          \
                                 {0.0f, 0.0f, 0.0f}}


/* for C only */
#define GLM_MAT3_IDENTITY ((mat3)GLM_MAT3_IDENTITY_INIT)
#define GLM_MAT3_ZERO     ((mat3)GLM_MAT3_ZERO_INIT)

/* DEPRECATED! use _copy, _ucopy versions */
#define glm_mat3_dup(mat, dest) glm_mat3_copy(mat, dest)

/*!
 * @brief copy all members of [mat] to [dest]
 *
 * @param[in]  mat  source
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_mat3_copy(mat3 mat, mat3 dest) {
  dest[0][0] = mat[0][0];
  dest[0][1] = mat[0][1];
  dest[0][2] = mat[0][2];

  dest[1][0] = mat[1][0];
  dest[1][1] = mat[1][1];
  dest[1][2] = mat[1][2];

  dest[2][0] = mat[2][0];
  dest[2][1] = mat[2][1];
  dest[2][2] = mat[2][2];
}

/*!
 * @brief make given matrix identity. It is identical with below,
 *        but it is more easy to do that with this func especially for members
 *        e.g. glm_mat3_identity(aStruct->aMatrix);
 *
 * @code
 * glm_mat3_copy(GLM_MAT3_IDENTITY, mat); // C only
 *
 * // or
 * mat3 mat = GLM_MAT3_IDENTITY_INIT;
 * @endcode
 *
 * @param[in, out]  mat  destination
 */
CGLM_INLINE
void
glm_mat3_identity(mat3 mat) {
  CGLM_ALIGN_MAT mat3 t = GLM_MAT3_IDENTITY_INIT;
  glm_mat3_copy(t, mat);
}

/*!
 * @brief make given matrix array's each element identity matrix
 *
 * @param[in, out]  mat   matrix array (must be aligned (16/32)
 *                        if alignment is not disabled)
 *
 * @param[in]       count count of matrices
 */
CGLM_INLINE
void
glm_mat3_identity_array(mat3 * __restrict mat, size_t count) {
  CGLM_ALIGN_MAT mat3 t = GLM_MAT3_IDENTITY_INIT;
  size_t i;

  for (i = 0; i < count; i++) {
    glm_mat3_copy(t, mat[i]);
  }
}

/*!
 * @brief make given matrix zero.
 *
 * @param[in, out]  mat  matrix
 */
CGLM_INLINE
void
glm_mat3_zero(mat3 mat) {
  CGLM_ALIGN_MAT mat3 t = GLM_MAT3_ZERO_INIT;
  glm_mat3_copy(t, mat);
}

/*!
 * @brief multiply m1 and m2 to dest
 *
 * m1, m2 and dest matrices can be same matrix, it is possible to write this:
 *
 * @code
 * mat3 m = GLM_MAT3_IDENTITY_INIT;
 * glm_mat3_mul(m, m, m);
 * @endcode
 *
 * @param[in]  m1   left matrix
 * @param[in]  m2   right matrix
 * @param[out] dest destination matrix
 */
CGLM_INLINE
void
glm_mat3_mul(mat3 m1, mat3 m2, mat3 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glm_mat3_mul_sse2(m1, m2, dest);
#else
  float a00 = m1[0][0], a01 = m1[0][1], a02 = m1[0][2],
        a10 = m1[1][0], a11 = m1[1][1], a12 = m1[1][2],
        a20 = m1[2][0], a21 = m1[2][1], a22 = m1[2][2],

        b00 = m2[0][0], b01 = m2[0][1], b02 = m2[0][2],
        b10 = m2[1][0], b11 = m2[1][1], b12 = m2[1][2],
        b20 = m2[2][0], b21 = m2[2][1], b22 = m2[2][2];

  dest[0][0] = a00 * b00 + a10 * b01 + a20 * b02;
  dest[0][1] = a01 * b00 + a11 * b01 + a21 * b02;
  dest[0][2] = a02 * b00 + a12 * b01 + a22 * b02;
  dest[1][0] = a00 * b10 + a10 * b11 + a20 * b12;
  dest[1][1] = a01 * b10 + a11 * b11 + a21 * b12;
  dest[1][2] = a02 * b10 + a12 * b11 + a22 * b12;
  dest[2][0] = a00 * b20 + a10 * b21 + a20 * b22;
  dest[2][1] = a01 * b20 + a11 * b21 + a21 * b22;
  dest[2][2] = a02 * b20 + a12 * b21 + a22 * b22;
#endif
}

/*!
 * @brief transpose mat3 and store in dest
 *
 * source matrix will not be transposed unless dest is m
 *
 * @param[in]  m     matrix
 * @param[out] dest  result
 */
CGLM_INLINE
void
glm_mat3_transpose_to(mat3 m, mat3 dest) {
  dest[0][0] = m[0][0];
  dest[0][1] = m[1][0];
  dest[0][2] = m[2][0];
  dest[1][0] = m[0][1];
  dest[1][1] = m[1][1];
  dest[1][2] = m[2][1];
  dest[2][0] = m[0][2];
  dest[2][1] = m[1][2];
  dest[2][2] = m[2][2];
}

/*!
 * @brief tranpose mat3 and store result in same matrix
 *
 * @param[in, out] m source and dest
 */
CGLM_INLINE
void
glm_mat3_transpose(mat3 m) {
  CGLM_ALIGN_MAT mat3 tmp;

  tmp[0][1] = m[1][0];
  tmp[0][2] = m[2][0];
  tmp[1][0] = m[0][1];
  tmp[1][2] = m[2][1];
  tmp[2][0] = m[0][2];
  tmp[2][1] = m[1][2];

  m[0][1] = tmp[0][1];
  m[0][2] = tmp[0][2];
  m[1][0] = tmp[1][0];
  m[1][2] = tmp[1][2];
  m[2][0] = tmp[2][0];
  m[2][1] = tmp[2][1];
}

/*!
 * @brief multiply mat3 with vec3 (column vector) and store in dest vector
 *
 * @param[in]  m    mat3 (left)
 * @param[in]  v    vec3 (right, column vector)
 * @param[out] dest vec3 (result, column vector)
 */
CGLM_INLINE
void
glm_mat3_mulv(mat3 m, vec3 v, vec3 dest) {
  vec3 res;
  res[0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2];
  res[1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2];
  res[2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2];
  glm_vec3_copy(res, dest);
}

/*!
 * @brief trace of matrix
 *
 * sum of the elements on the main diagonal from upper left to the lower right
 *
 * @param[in]  m matrix
 */
CGLM_INLINE
float
glm_mat3_trace(mat3 m) {
  return m[0][0] + m[1][1] + m[2][2];
}

/*!
 * @brief convert mat3 to quaternion
 *
 * @param[in]  m    rotation matrix
 * @param[out] dest destination quaternion
 */
CGLM_INLINE
void
glm_mat3_quat(mat3 m, versor dest) {
  float trace, r, rinv;

  /* it seems using like m12 instead of m[1][2] causes extra instructions */

  trace = m[0][0] + m[1][1] + m[2][2];
  if (trace >= 0.0f) {
    r       = sqrtf(1.0f + trace);
    rinv    = 0.5f / r;

    dest[0] = rinv * (m[1][2] - m[2][1]);
    dest[1] = rinv * (m[2][0] - m[0][2]);
    dest[2] = rinv * (m[0][1] - m[1][0]);
    dest[3] = r    * 0.5f;
  } else if (m[0][0] >= m[1][1] && m[0][0] >= m[2][2]) {
    r       = sqrtf(1.0f - m[1][1] - m[2][2] + m[0][0]);
    rinv    = 0.5f / r;

    dest[0] = r    * 0.5f;
    dest[1] = rinv * (m[0][1] + m[1][0]);
    dest[2] = rinv * (m[0][2] + m[2][0]);
    dest[3] = rinv * (m[1][2] - m[2][1]);
  } else if (m[1][1] >= m[2][2]) {
    r       = sqrtf(1.0f - m[0][0] - m[2][2] + m[1][1]);
    rinv    = 0.5f / r;

    dest[0] = rinv * (m[0][1] + m[1][0]);
    dest[1] = r    * 0.5f;
    dest[2] = rinv * (m[1][2] + m[2][1]);
    dest[3] = rinv * (m[2][0] - m[0][2]);
  } else {
    r       = sqrtf(1.0f - m[0][0] - m[1][1] + m[2][2]);
    rinv    = 0.5f / r;

    dest[0] = rinv * (m[0][2] + m[2][0]);
    dest[1] = rinv * (m[1][2] + m[2][1]);
    dest[2] = r    * 0.5f;
    dest[3] = rinv * (m[0][1] - m[1][0]);
  }
}

/*!
 * @brief scale (multiply with scalar) matrix
 *
 * multiply matrix with scalar
 *
 * @param[in, out] m matrix
 * @param[in]      s scalar
 */
CGLM_INLINE
void
glm_mat3_scale(mat3 m, float s) {
  m[0][0] *= s; m[0][1] *= s; m[0][2] *= s;
  m[1][0] *= s; m[1][1] *= s; m[1][2] *= s;
  m[2][0] *= s; m[2][1] *= s; m[2][2] *= s;
}

/*!
 * @brief mat3 determinant
 *
 * @param[in] mat matrix
 *
 * @return determinant
 */
CGLM_INLINE
float
glm_mat3_det(mat3 mat) {
  float a = mat[0][0], b = mat[0][1], c = mat[0][2],
        d = mat[1][0], e = mat[1][1], f = mat[1][2],
        g = mat[2][0], h = mat[2][1], i = mat[2][2];

  return a * (e * i - h * f) - d * (b * i - c * h) + g * (b * f - c * e);
}

/*!
 * @brief inverse mat3 and store in dest
 *
 * @param[in]  mat  matrix
 * @param[out] dest inverse matrix
 */
CGLM_INLINE
void
glm_mat3_inv(mat3 mat, mat3 dest) {
  float det;
  float a = mat[0][0], b = mat[0][1], c = mat[0][2],
        d = mat[1][0], e = mat[1][1], f = mat[1][2],
        g = mat[2][0], h = mat[2][1], i = mat[2][2];

  dest[0][0] =   e * i - f * h;
  dest[0][1] = -(b * i - h * c);
  dest[0][2] =   b * f - e * c;
  dest[1][0] = -(d * i - g * f);
  dest[1][1] =   a * i - c * g;
  dest[1][2] = -(a * f - d * c);
  dest[2][0] =   d * h - g * e;
  dest[2][1] = -(a * h - g * b);
  dest[2][2] =   a * e - b * d;

  det = 1.0f / (a * dest[0][0] + b * dest[1][0] + c * dest[2][0]);

  glm_mat3_scale(dest, det);
}

/*!
 * @brief swap two matrix columns
 *
 * @param[in,out] mat  matrix
 * @param[in]     col1 col1
 * @param[in]     col2 col2
 */
CGLM_INLINE
void
glm_mat3_swap_col(mat3 mat, int col1, int col2) {
  vec3 tmp;
  glm_vec3_copy(mat[col1], tmp);
  glm_vec3_copy(mat[col2], mat[col1]);
  glm_vec3_copy(tmp, mat[col2]);
}

/*!
 * @brief swap two matrix rows
 *
 * @param[in,out] mat  matrix
 * @param[in]     row1 row1
 * @param[in]     row2 row2
 */
CGLM_INLINE
void
glm_mat3_swap_row(mat3 mat, int row1, int row2) {
  vec3 tmp;
  tmp[0] = mat[0][row1];
  tmp[1] = mat[1][row1];
  tmp[2] = mat[2][row1];

  mat[0][row1] = mat[0][row2];
  mat[1][row1] = mat[1][row2];
  mat[2][row1] = mat[2][row2];

  mat[0][row2] = tmp[0];
  mat[1][row2] = tmp[1];
  mat[2][row2] = tmp[2];
}

/*!
 * @brief helper for  R (row vector) * M (matrix) * C (column vector)
 *
 * rmc stands for Row * Matrix * Column
 *
 * the result is scalar because R * M = Matrix1x3 (row vector),
 * then Matrix1x3 * Vec3 (column vector) = Matrix1x1 (Scalar)
 *
 * @param[in]  r   row vector or matrix1x3
 * @param[in]  m   matrix3x3
 * @param[in]  c   column vector or matrix3x1
 *
 * @return scalar value e.g. Matrix1x1
 */
CGLM_INLINE
float
glm_mat3_rmc(vec3 r, mat3 m, vec3 c) {
  vec3 tmp;
  glm_mat3_mulv(m, c, tmp);
  return glm_vec3_dot(r, tmp);
}

#endif /* cglm_mat3_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*
 Macros:
   GLM_MAT2_IDENTITY_INIT
   GLM_MAT2_ZERO_INIT
   GLM_MAT2_IDENTITY
   GLM_MAT2_ZERO

 Functions:
   CGLM_INLINE void  glm_mat2_copy(mat2 mat, mat2 dest)
   CGLM_INLINE void  glm_mat2_identity(mat2 mat)
   CGLM_INLINE void  glm_mat2_identity_array(mat2 * restrict mat, size_t count)
   CGLM_INLINE void  glm_mat2_zero(mat2 mat)
   CGLM_INLINE void  glm_mat2_mul(mat2 m1, mat2 m2, mat2 dest)
   CGLM_INLINE void  glm_mat2_transpose_to(mat2 m, mat2 dest)
   CGLM_INLINE void  glm_mat2_transpose(mat2 m)
   CGLM_INLINE void  glm_mat2_mulv(mat2 m, vec2 v, vec2 dest)
   CGLM_INLINE float glm_mat2_trace(mat2 m)
   CGLM_INLINE void  glm_mat2_scale(mat2 m, float s)
   CGLM_INLINE float glm_mat2_det(mat2 mat)
   CGLM_INLINE void  glm_mat2_inv(mat2 mat, mat2 dest)
   CGLM_INLINE void  glm_mat2_swap_col(mat2 mat, int col1, int col2)
   CGLM_INLINE void  glm_mat2_swap_row(mat2 mat, int row1, int row2)
   CGLM_INLINE float glm_mat2_rmc(vec2 r, mat2 m, vec2 c)
 */

#ifndef cglm_mat2_h
#define cglm_mat2_h


#ifdef CGLM_SSE_FP
/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_mat2_sse_h
#define cglm_mat2_sse_h
#if defined( __SSE__ ) || defined( __SSE2__ )


CGLM_INLINE
void
glm_mat2_mul_sse2(mat2 m1, mat2 m2, mat2 dest) {
  __m128 x0, x1, x2;

  x1 = glmm_load(m1[0]); /* d c b a */
  x2 = glmm_load(m2[0]); /* h g f e */

  /*
   dest[0][0] = a * e + c * f;
   dest[0][1] = b * e + d * f;
   dest[1][0] = a * g + c * h;
   dest[1][1] = b * g + d * h;
   */
  x0 = _mm_mul_ps(_mm_movelh_ps(x1, x1), glmm_shuff1(x2, 2, 2, 0, 0));
  x1 = _mm_mul_ps(_mm_movehl_ps(x1, x1), glmm_shuff1(x2, 3, 3, 1, 1));
  x1 = _mm_add_ps(x0, x1);

  glmm_store(dest[0], x1);
}

CGLM_INLINE
void
glm_mat2_transp_sse2(mat2 m, mat2 dest) {
  /* d c b a */
  /* d b c a */
  glmm_store(dest[0], glmm_shuff1(glmm_load(m[0]), 3, 1, 2, 0));
}

#endif
#endif /* cglm_mat2_sse_h */

#endif

#define GLM_MAT2_IDENTITY_INIT  {{1.0f, 0.0f}, {0.0f, 1.0f}}
#define GLM_MAT2_ZERO_INIT      {{0.0f, 0.0f}, {0.0f, 0.0f}}

/* for C only */
#define GLM_MAT2_IDENTITY ((mat2)GLM_MAT2_IDENTITY_INIT)
#define GLM_MAT2_ZERO     ((mat2)GLM_MAT2_ZERO_INIT)

/*!
 * @brief copy all members of [mat] to [dest]
 *
 * @param[in]  mat  source
 * @param[out] dest destination
 */
CGLM_INLINE
void
glm_mat2_copy(mat2 mat, mat2 dest) {
  glm_vec4_ucopy(mat[0], dest[0]);
}

/*!
 * @brief make given matrix identity. It is identical with below,
 *        but it is more easy to do that with this func especially for members
 *        e.g. glm_mat2_identity(aStruct->aMatrix);
 *
 * @code
 * glm_mat2_copy(GLM_MAT2_IDENTITY, mat); // C only
 *
 * // or
 * mat2 mat = GLM_MAT2_IDENTITY_INIT;
 * @endcode
 *
 * @param[in, out]  mat  destination
 */
CGLM_INLINE
void
glm_mat2_identity(mat2 mat) {
  CGLM_ALIGN_MAT mat2 t = GLM_MAT2_IDENTITY_INIT;
  glm_mat2_copy(t, mat);
}

/*!
 * @brief make given matrix array's each element identity matrix
 *
 * @param[in, out]  mat   matrix array (must be aligned (16)
 *                        if alignment is not disabled)
 *
 * @param[in]       count count of matrices
 */
CGLM_INLINE
void
glm_mat2_identity_array(mat2 * __restrict mat, size_t count) {
  CGLM_ALIGN_MAT mat2 t = GLM_MAT2_IDENTITY_INIT;
  size_t i;

  for (i = 0; i < count; i++) {
    glm_mat2_copy(t, mat[i]);
  }
}

/*!
 * @brief make given matrix zero.
 *
 * @param[in, out]  mat  matrix
 */
CGLM_INLINE
void
glm_mat2_zero(mat2 mat) {
  CGLM_ALIGN_MAT mat2 t = GLM_MAT2_ZERO_INIT;
  glm_mat2_copy(t, mat);
}

/*!
 * @brief multiply m1 and m2 to dest
 *
 * m1, m2 and dest matrices can be same matrix, it is possible to write this:
 *
 * @code
 * mat2 m = GLM_MAT2_IDENTITY_INIT;
 * glm_mat2_mul(m, m, m);
 * @endcode
 *
 * @param[in]  m1   left matrix
 * @param[in]  m2   right matrix
 * @param[out] dest destination matrix
 */
CGLM_INLINE
void
glm_mat2_mul(mat2 m1, mat2 m2, mat2 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glm_mat2_mul_sse2(m1, m2, dest);
#else
  float a00 = m1[0][0], a01 = m1[0][1],
        a10 = m1[1][0], a11 = m1[1][1],
        b00 = m2[0][0], b01 = m2[0][1],
        b10 = m2[1][0], b11 = m2[1][1];

  dest[0][0] = a00 * b00 + a10 * b01;
  dest[0][1] = a01 * b00 + a11 * b01;
  dest[1][0] = a00 * b10 + a10 * b11;
  dest[1][1] = a01 * b10 + a11 * b11;
#endif
}

/*!
 * @brief transpose mat2 and store in dest
 *
 * source matrix will not be transposed unless dest is m
 *
 * @param[in]  m     matrix
 * @param[out] dest  result
 */
CGLM_INLINE
void
glm_mat2_transpose_to(mat2 m, mat2 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glm_mat2_transp_sse2(m, dest);
#else
  dest[0][0] = m[0][0];
  dest[0][1] = m[1][0];
  dest[1][0] = m[0][1];
  dest[1][1] = m[1][1];
#endif
}

/*!
 * @brief tranpose mat2 and store result in same matrix
 *
 * @param[in, out] m source and dest
 */
CGLM_INLINE
void
glm_mat2_transpose(mat2 m) {
  float tmp;
  tmp     = m[0][1];
  m[0][1] = m[1][0];
  m[1][0] = tmp;
}

/*!
 * @brief multiply mat2 with vec2 (column vector) and store in dest vector
 *
 * @param[in]  m    mat2 (left)
 * @param[in]  v    vec2 (right, column vector)
 * @param[out] dest vec2 (result, column vector)
 */
CGLM_INLINE
void
glm_mat2_mulv(mat2 m, vec2 v, vec2 dest) {
  dest[0] = m[0][0] * v[0] + m[1][0] * v[1];
  dest[1] = m[0][1] * v[0] + m[1][1] * v[1];
}

/*!
 * @brief trace of matrix
 *
 * sum of the elements on the main diagonal from upper left to the lower right
 *
 * @param[in]  m matrix
 */
CGLM_INLINE
float
glm_mat2_trace(mat2 m) {
  return m[0][0] + m[1][1];
}

/*!
 * @brief scale (multiply with scalar) matrix
 *
 * multiply matrix with scalar
 *
 * @param[in, out] m matrix
 * @param[in]      s scalar
 */
CGLM_INLINE
void
glm_mat2_scale(mat2 m, float s) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(m[0], _mm_mul_ps(_mm_loadu_ps(m[0]), _mm_set1_ps(s)));
#elif defined(CGLM_NEON_FP)
  vst1q_f32(m[0], vmulq_f32(vld1q_f32(m[0]), vdupq_n_f32(s)));
#else
  m[0][0] = m[0][0] * s;
  m[0][1] = m[0][1] * s;
  m[1][0] = m[1][0] * s;
  m[1][1] = m[1][1] * s;
#endif
}

/*!
 * @brief mat2 determinant
 *
 * @param[in] mat matrix
 *
 * @return determinant
 */
CGLM_INLINE
float
glm_mat2_det(mat2 mat) {
  return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
}

/*!
 * @brief inverse mat2 and store in dest
 *
 * @param[in]  mat  matrix
 * @param[out] dest inverse matrix
 */
CGLM_INLINE
void
glm_mat2_inv(mat2 mat, mat2 dest) {
  float det;
  float a = mat[0][0], b = mat[0][1],
        c = mat[1][0], d = mat[1][1];

  det = 1.0f / (a * d - b * c);

  dest[0][0] =  d * det;
  dest[0][1] = -b * det;
  dest[1][0] = -c * det;
  dest[1][1] =  a * det;
}

/*!
 * @brief swap two matrix columns
 *
 * @param[in,out] mat  matrix
 * @param[in]     col1 col1
 * @param[in]     col2 col2
 */
CGLM_INLINE
void
glm_mat2_swap_col(mat2 mat, int col1, int col2) {
  float a, b;

  a = mat[col1][0];
  b = mat[col1][1];

  mat[col1][0] = mat[col2][0];
  mat[col1][1] = mat[col2][1];

  mat[col2][0] = a;
  mat[col2][1] = b;
}

/*!
 * @brief swap two matrix rows
 *
 * @param[in,out] mat  matrix
 * @param[in]     row1 row1
 * @param[in]     row2 row2
 */
CGLM_INLINE
void
glm_mat2_swap_row(mat2 mat, int row1, int row2) {
  float a, b;

  a = mat[0][row1];
  b = mat[1][row1];

  mat[0][row1] = mat[0][row2];
  mat[1][row1] = mat[1][row2];

  mat[0][row2] = a;
  mat[1][row2] = b;
}

/*!
 * @brief helper for  R (row vector) * M (matrix) * C (column vector)
 *
 * rmc stands for Row * Matrix * Column
 *
 * the result is scalar because R * M = Matrix1x2 (row vector),
 * then Matrix1x2 * Vec2 (column vector) = Matrix1x1 (Scalar)
 *
 * @param[in]  r   row vector or matrix1x2
 * @param[in]  m   matrix2x2
 * @param[in]  c   column vector or matrix2x1
 *
 * @return scalar value e.g. Matrix1x1
 */
CGLM_INLINE
float
glm_mat2_rmc(vec2 r, mat2 m, vec2 c) {
  vec2 tmp;
  glm_mat2_mulv(m, c, tmp);
  return glm_vec2_dot(r, tmp);
}

#endif /* cglm_mat2_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*
 Functions:
   CGLM_INLINE void glm_translate_to(mat4 m, vec3 v, mat4 dest);
   CGLM_INLINE void glm_translate(mat4 m, vec3 v);
   CGLM_INLINE void glm_translate_x(mat4 m, float to);
   CGLM_INLINE void glm_translate_y(mat4 m, float to);
   CGLM_INLINE void glm_translate_z(mat4 m, float to);
   CGLM_INLINE void glm_translate_make(mat4 m, vec3 v);
   CGLM_INLINE void glm_scale_to(mat4 m, vec3 v, mat4 dest);
   CGLM_INLINE void glm_scale_make(mat4 m, vec3 v);
   CGLM_INLINE void glm_scale(mat4 m, vec3 v);
   CGLM_INLINE void glm_scale_uni(mat4 m, float s);
   CGLM_INLINE void glm_rotate_x(mat4 m, float angle, mat4 dest);
   CGLM_INLINE void glm_rotate_y(mat4 m, float angle, mat4 dest);
   CGLM_INLINE void glm_rotate_z(mat4 m, float angle, mat4 dest);
   CGLM_INLINE void glm_rotate_make(mat4 m, float angle, vec3 axis);
   CGLM_INLINE void glm_rotate(mat4 m, float angle, vec3 axis);
   CGLM_INLINE void glm_rotate_at(mat4 m, vec3 pivot, float angle, vec3 axis);
   CGLM_INLINE void glm_rotate_atm(mat4 m, vec3 pivot, float angle, vec3 axis);
   CGLM_INLINE void glm_decompose_scalev(mat4 m, vec3 s);
   CGLM_INLINE bool glm_uniscaled(mat4 m);
   CGLM_INLINE void glm_decompose_rs(mat4 m, mat4 r, vec3 s);
   CGLM_INLINE void glm_decompose(mat4 m, vec4 t, mat4 r, vec3 s);
 */

#ifndef cglm_affine_h
#define cglm_affine_h

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*
 Functions:
   CGLM_INLINE void glm_mul(mat4 m1, mat4 m2, mat4 dest);
   CGLM_INLINE void glm_inv_tr(mat4 mat);
 */

#ifndef cglm_affine_mat_h
#define cglm_affine_mat_h


#ifdef CGLM_SSE_FP
/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_affine_mat_sse2_h
#define cglm_affine_mat_sse2_h
#if defined( __SSE__ ) || defined( __SSE2__ )


CGLM_INLINE
void
glm_mul_sse2(mat4 m1, mat4 m2, mat4 dest) {
  /* D = R * L (Column-Major) */
  __m128 l0, l1, l2, l3, r;

  l0 = glmm_load(m1[0]);
  l1 = glmm_load(m1[1]);
  l2 = glmm_load(m1[2]);
  l3 = glmm_load(m1[3]);

  r = glmm_load(m2[0]);
  glmm_store(dest[0],
             _mm_add_ps(_mm_add_ps(_mm_mul_ps(glmm_shuff1x(r, 0), l0),
                                   _mm_mul_ps(glmm_shuff1x(r, 1), l1)),
                        _mm_mul_ps(glmm_shuff1x(r, 2), l2)));

  r = glmm_load(m2[1]);
  glmm_store(dest[1],
             _mm_add_ps(_mm_add_ps(_mm_mul_ps(glmm_shuff1x(r, 0), l0),
                                   _mm_mul_ps(glmm_shuff1x(r, 1), l1)),
                        _mm_mul_ps(glmm_shuff1x(r, 2), l2)));

  r = glmm_load(m2[2]);
  glmm_store(dest[2],
             _mm_add_ps(_mm_add_ps(_mm_mul_ps(glmm_shuff1x(r, 0), l0),
                                   _mm_mul_ps(glmm_shuff1x(r, 1), l1)),
                        _mm_mul_ps(glmm_shuff1x(r, 2), l2)));

  r = glmm_load(m2[3]);
  glmm_store(dest[3],
             _mm_add_ps(_mm_add_ps(_mm_mul_ps(glmm_shuff1x(r, 0), l0),
                                   _mm_mul_ps(glmm_shuff1x(r, 1), l1)),
                        _mm_add_ps(_mm_mul_ps(glmm_shuff1x(r, 2), l2),
                                   _mm_mul_ps(glmm_shuff1x(r, 3), l3))));
}

CGLM_INLINE
void
glm_mul_rot_sse2(mat4 m1, mat4 m2, mat4 dest) {
  /* D = R * L (Column-Major) */
  __m128 l0, l1, l2, l3, r;

  l0 = glmm_load(m1[0]);
  l1 = glmm_load(m1[1]);
  l2 = glmm_load(m1[2]);
  l3 = glmm_load(m1[3]);

  r = glmm_load(m2[0]);
  glmm_store(dest[0],
             _mm_add_ps(_mm_add_ps(_mm_mul_ps(glmm_shuff1x(r, 0), l0),
                                   _mm_mul_ps(glmm_shuff1x(r, 1), l1)),
                        _mm_mul_ps(glmm_shuff1x(r, 2), l2)));

  r = glmm_load(m2[1]);
  glmm_store(dest[1],
             _mm_add_ps(_mm_add_ps(_mm_mul_ps(glmm_shuff1x(r, 0), l0),
                                   _mm_mul_ps(glmm_shuff1x(r, 1), l1)),
                        _mm_mul_ps(glmm_shuff1x(r, 2), l2)));

  r = glmm_load(m2[2]);
  glmm_store(dest[2],
             _mm_add_ps(_mm_add_ps(_mm_mul_ps(glmm_shuff1x(r, 0), l0),
                                   _mm_mul_ps(glmm_shuff1x(r, 1), l1)),
                        _mm_mul_ps(glmm_shuff1x(r, 2), l2)));

  glmm_store(dest[3], l3);
}

CGLM_INLINE
void
glm_inv_tr_sse2(mat4 mat) {
  __m128 r0, r1, r2, r3, x0, x1;

  r0 = glmm_load(mat[0]);
  r1 = glmm_load(mat[1]);
  r2 = glmm_load(mat[2]);
  r3 = glmm_load(mat[3]);
  x1 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);

  _MM_TRANSPOSE4_PS(r0, r1, r2, x1);

  x0 = _mm_add_ps(_mm_mul_ps(r0, glmm_shuff1(r3, 0, 0, 0, 0)),
                  _mm_mul_ps(r1, glmm_shuff1(r3, 1, 1, 1, 1)));
  x0 = _mm_add_ps(x0, _mm_mul_ps(r2, glmm_shuff1(r3, 2, 2, 2, 2)));
  x0 = _mm_xor_ps(x0, _mm_set1_ps(-0.f));

  x0 = _mm_add_ps(x0, x1);

  glmm_store(mat[0], r0);
  glmm_store(mat[1], r1);
  glmm_store(mat[2], r2);
  glmm_store(mat[3], x0);
}

#endif
#endif /* cglm_affine_mat_sse2_h */

#endif

#ifdef CGLM_AVX_FP
/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_affine_mat_avx_h
#define cglm_affine_mat_avx_h
#ifdef __AVX__


#include <immintrin.h>

CGLM_INLINE
void
glm_mul_avx(mat4 m1, mat4 m2, mat4 dest) {
  /* D = R * L (Column-Major) */

  __m256 y0, y1, y2, y3, y4, y5, y6, y7, y8, y9;

  y0 = glmm_load256(m2[0]); /* h g f e d c b a */
  y1 = glmm_load256(m2[2]); /* p o n m l k j i */

  y2 = glmm_load256(m1[0]); /* h g f e d c b a */
  y3 = glmm_load256(m1[2]); /* p o n m l k j i */

  /* 0x03: 0b00000011 */
  y4 = _mm256_permute2f128_ps(y2, y2, 0x03); /* d c b a h g f e */
  y5 = _mm256_permute2f128_ps(y3, y3, 0x03); /* l k j i p o n m */

  /* f f f f a a a a */
  /* h h h h c c c c */
  /* e e e e b b b b */
  /* g g g g d d d d */
  y6 = _mm256_permutevar_ps(y0, _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0));
  y7 = _mm256_permutevar_ps(y0, _mm256_set_epi32(3, 3, 3, 3, 2, 2, 2, 2));
  y8 = _mm256_permutevar_ps(y0, _mm256_set_epi32(0, 0, 0, 0, 1, 1, 1, 1));
  y9 = _mm256_permutevar_ps(y0, _mm256_set_epi32(2, 2, 2, 2, 3, 3, 3, 3));

  glmm_store256(dest[0],
                _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(y2, y6),
                                            _mm256_mul_ps(y3, y7)),
                              _mm256_add_ps(_mm256_mul_ps(y4, y8),
                                            _mm256_mul_ps(y5, y9))));

  /* n n n n i i i i */
  /* p p p p k k k k */
  /* m m m m j j j j */
  /* o o o o l l l l */
  y6 = _mm256_permutevar_ps(y1, _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0));
  y7 = _mm256_permutevar_ps(y1, _mm256_set_epi32(3, 3, 3, 3, 2, 2, 2, 2));
  y8 = _mm256_permutevar_ps(y1, _mm256_set_epi32(0, 0, 0, 0, 1, 1, 1, 1));
  y9 = _mm256_permutevar_ps(y1, _mm256_set_epi32(2, 2, 2, 2, 3, 3, 3, 3));

  glmm_store256(dest[2],
                _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(y2, y6),
                                            _mm256_mul_ps(y3, y7)),
                              _mm256_add_ps(_mm256_mul_ps(y4, y8),
                                            _mm256_mul_ps(y5, y9))));
}

#endif
#endif /* cglm_affine_mat_avx_h */

#endif

/*!
 * @brief this is similar to glm_mat4_mul but specialized to affine transform
 *
 * Matrix format should be:
 *   R  R  R  X
 *   R  R  R  Y
 *   R  R  R  Z
 *   0  0  0  W
 *
 * this reduces some multiplications. It should be faster than mat4_mul.
 * if you are not sure about matrix format then DON'T use this! use mat4_mul
 *
 * @param[in]   m1    affine matrix 1
 * @param[in]   m2    affine matrix 2
 * @param[out]  dest  result matrix
 */
CGLM_INLINE
void
glm_mul(mat4 m1, mat4 m2, mat4 dest) {
#ifdef __AVX__
  glm_mul_avx(m1, m2, dest);
#elif defined( __SSE__ ) || defined( __SSE2__ )
  glm_mul_sse2(m1, m2, dest);
#else
  float a00 = m1[0][0], a01 = m1[0][1], a02 = m1[0][2], a03 = m1[0][3],
        a10 = m1[1][0], a11 = m1[1][1], a12 = m1[1][2], a13 = m1[1][3],
        a20 = m1[2][0], a21 = m1[2][1], a22 = m1[2][2], a23 = m1[2][3],
        a30 = m1[3][0], a31 = m1[3][1], a32 = m1[3][2], a33 = m1[3][3],

        b00 = m2[0][0], b01 = m2[0][1], b02 = m2[0][2],
        b10 = m2[1][0], b11 = m2[1][1], b12 = m2[1][2],
        b20 = m2[2][0], b21 = m2[2][1], b22 = m2[2][2],
        b30 = m2[3][0], b31 = m2[3][1], b32 = m2[3][2], b33 = m2[3][3];

  dest[0][0] = a00 * b00 + a10 * b01 + a20 * b02;
  dest[0][1] = a01 * b00 + a11 * b01 + a21 * b02;
  dest[0][2] = a02 * b00 + a12 * b01 + a22 * b02;
  dest[0][3] = a03 * b00 + a13 * b01 + a23 * b02;

  dest[1][0] = a00 * b10 + a10 * b11 + a20 * b12;
  dest[1][1] = a01 * b10 + a11 * b11 + a21 * b12;
  dest[1][2] = a02 * b10 + a12 * b11 + a22 * b12;
  dest[1][3] = a03 * b10 + a13 * b11 + a23 * b12;

  dest[2][0] = a00 * b20 + a10 * b21 + a20 * b22;
  dest[2][1] = a01 * b20 + a11 * b21 + a21 * b22;
  dest[2][2] = a02 * b20 + a12 * b21 + a22 * b22;
  dest[2][3] = a03 * b20 + a13 * b21 + a23 * b22;

  dest[3][0] = a00 * b30 + a10 * b31 + a20 * b32 + a30 * b33;
  dest[3][1] = a01 * b30 + a11 * b31 + a21 * b32 + a31 * b33;
  dest[3][2] = a02 * b30 + a12 * b31 + a22 * b32 + a32 * b33;
  dest[3][3] = a03 * b30 + a13 * b31 + a23 * b32 + a33 * b33;
#endif
}

/*!
 * @brief this is similar to glm_mat4_mul but specialized to affine transform
 *
 * Right Matrix format should be:
 *   R  R  R  0
 *   R  R  R  0
 *   R  R  R  0
 *   0  0  0  1
 *
 * this reduces some multiplications. It should be faster than mat4_mul.
 * if you are not sure about matrix format then DON'T use this! use mat4_mul
 *
 * @param[in]   m1    affine matrix 1
 * @param[in]   m2    affine matrix 2
 * @param[out]  dest  result matrix
 */
CGLM_INLINE
void
glm_mul_rot(mat4 m1, mat4 m2, mat4 dest) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glm_mul_rot_sse2(m1, m2, dest);
#else
  float a00 = m1[0][0], a01 = m1[0][1], a02 = m1[0][2], a03 = m1[0][3],
        a10 = m1[1][0], a11 = m1[1][1], a12 = m1[1][2], a13 = m1[1][3],
        a20 = m1[2][0], a21 = m1[2][1], a22 = m1[2][2], a23 = m1[2][3],
        a30 = m1[3][0], a31 = m1[3][1], a32 = m1[3][2], a33 = m1[3][3],

        b00 = m2[0][0], b01 = m2[0][1], b02 = m2[0][2],
        b10 = m2[1][0], b11 = m2[1][1], b12 = m2[1][2],
        b20 = m2[2][0], b21 = m2[2][1], b22 = m2[2][2];

  dest[0][0] = a00 * b00 + a10 * b01 + a20 * b02;
  dest[0][1] = a01 * b00 + a11 * b01 + a21 * b02;
  dest[0][2] = a02 * b00 + a12 * b01 + a22 * b02;
  dest[0][3] = a03 * b00 + a13 * b01 + a23 * b02;

  dest[1][0] = a00 * b10 + a10 * b11 + a20 * b12;
  dest[1][1] = a01 * b10 + a11 * b11 + a21 * b12;
  dest[1][2] = a02 * b10 + a12 * b11 + a22 * b12;
  dest[1][3] = a03 * b10 + a13 * b11 + a23 * b12;

  dest[2][0] = a00 * b20 + a10 * b21 + a20 * b22;
  dest[2][1] = a01 * b20 + a11 * b21 + a21 * b22;
  dest[2][2] = a02 * b20 + a12 * b21 + a22 * b22;
  dest[2][3] = a03 * b20 + a13 * b21 + a23 * b22;

  dest[3][0] = a30;
  dest[3][1] = a31;
  dest[3][2] = a32;
  dest[3][3] = a33;
#endif
}

/*!
 * @brief inverse orthonormal rotation + translation matrix (ridig-body)
 *
 * @code
 * X = | R  T |   X' = | R' -R'T |
 *     | 0  1 |        | 0     1 |
 * @endcode
 *
 * @param[in,out]  mat  matrix
 */
CGLM_INLINE
void
glm_inv_tr(mat4 mat) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glm_inv_tr_sse2(mat);
#else
  CGLM_ALIGN_MAT mat3 r;
  CGLM_ALIGN(8)  vec3 t;

  /* rotate */
  glm_mat4_pick3t(mat, r);
  glm_mat4_ins3(r, mat);

  /* translate */
  glm_mat3_mulv(r, mat[3], t);
  glm_vec3_negate(t);
  glm_vec3_copy(t, mat[3]);
#endif
}

#endif /* cglm_affine_mat_h */


/*!
 * @brief translate existing transform matrix by v vector
 *        and stores result in same matrix
 *
 * @param[in, out]  m  affine transfrom
 * @param[in]       v  translate vector [x, y, z]
 */
CGLM_INLINE
void
glm_translate(mat4 m, vec3 v) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(m[3],
             _mm_add_ps(_mm_add_ps(_mm_mul_ps(glmm_load(m[0]),
                                              _mm_set1_ps(v[0])),
                                   _mm_mul_ps(glmm_load(m[1]),
                                              _mm_set1_ps(v[1]))),
                        _mm_add_ps(_mm_mul_ps(glmm_load(m[2]),
                                              _mm_set1_ps(v[2])),
                                   glmm_load(m[3]))))
  ;
#else
  vec4 v1, v2, v3;

  glm_vec4_scale(m[0], v[0], v1);
  glm_vec4_scale(m[1], v[1], v2);
  glm_vec4_scale(m[2], v[2], v3);

  glm_vec4_add(v1, m[3], m[3]);
  glm_vec4_add(v2, m[3], m[3]);
  glm_vec4_add(v3, m[3], m[3]);
#endif
}

/*!
 * @brief translate existing transform matrix by v vector
 *        and store result in dest
 *
 * source matrix will remain same
 *
 * @param[in]  m    affine transfrom
 * @param[in]  v    translate vector [x, y, z]
 * @param[out] dest translated matrix
 */
CGLM_INLINE
void
glm_translate_to(mat4 m, vec3 v, mat4 dest) {
  glm_mat4_copy(m, dest);
  glm_translate(dest, v);
}

/*!
 * @brief translate existing transform matrix by x factor
 *
 * @param[in, out]  m  affine transfrom
 * @param[in]       x  x factor
 */
CGLM_INLINE
void
glm_translate_x(mat4 m, float x) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(m[3],
             _mm_add_ps(_mm_mul_ps(glmm_load(m[0]),
                                   _mm_set1_ps(x)),
                        glmm_load(m[3])))
  ;
#else
  vec4 v1;
  glm_vec4_scale(m[0], x, v1);
  glm_vec4_add(v1, m[3], m[3]);
#endif
}

/*!
 * @brief translate existing transform matrix by y factor
 *
 * @param[in, out]  m  affine transfrom
 * @param[in]       y  y factor
 */
CGLM_INLINE
void
glm_translate_y(mat4 m, float y) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(m[3],
             _mm_add_ps(_mm_mul_ps(glmm_load(m[1]),
                                   _mm_set1_ps(y)),
                        glmm_load(m[3])))
  ;
#else
  vec4 v1;
  glm_vec4_scale(m[1], y, v1);
  glm_vec4_add(v1, m[3], m[3]);
#endif
}

/*!
 * @brief translate existing transform matrix by z factor
 *
 * @param[in, out]  m  affine transfrom
 * @param[in]       z  z factor
 */
CGLM_INLINE
void
glm_translate_z(mat4 m, float z) {
#if defined( __SSE__ ) || defined( __SSE2__ )
  glmm_store(m[3],
             _mm_add_ps(_mm_mul_ps(glmm_load(m[2]),
                                   _mm_set1_ps(z)),
                        glmm_load(m[3])))
  ;
#else
  vec4 v1;
  glm_vec4_scale(m[2], z, v1);
  glm_vec4_add(v1, m[3], m[3]);
#endif
}

/*!
 * @brief creates NEW translate transform matrix by v vector
 *
 * @param[out]  m  affine transfrom
 * @param[in]   v  translate vector [x, y, z]
 */
CGLM_INLINE
void
glm_translate_make(mat4 m, vec3 v) {
  glm_mat4_identity(m);
  glm_vec3_copy(v, m[3]);
}

/*!
 * @brief scale existing transform matrix by v vector
 *        and store result in dest
 *
 * @param[in]  m    affine transfrom
 * @param[in]  v    scale vector [x, y, z]
 * @param[out] dest scaled matrix
 */
CGLM_INLINE
void
glm_scale_to(mat4 m, vec3 v, mat4 dest) {
  glm_vec4_scale(m[0], v[0], dest[0]);
  glm_vec4_scale(m[1], v[1], dest[1]);
  glm_vec4_scale(m[2], v[2], dest[2]);

  glm_vec4_copy(m[3], dest[3]);
}

/*!
 * @brief creates NEW scale matrix by v vector
 *
 * @param[out]  m  affine transfrom
 * @param[in]   v  scale vector [x, y, z]
 */
CGLM_INLINE
void
glm_scale_make(mat4 m, vec3 v) {
  glm_mat4_identity(m);
  m[0][0] = v[0];
  m[1][1] = v[1];
  m[2][2] = v[2];
}

/*!
 * @brief scales existing transform matrix by v vector
 *        and stores result in same matrix
 *
 * @param[in, out]  m  affine transfrom
 * @param[in]       v  scale vector [x, y, z]
 */
CGLM_INLINE
void
glm_scale(mat4 m, vec3 v) {
  glm_scale_to(m, v, m);
}

/*!
 * @brief applies uniform scale to existing transform matrix v = [s, s, s]
 *        and stores result in same matrix
 *
 * @param[in, out]  m  affine transfrom
 * @param[in]       s  scale factor
 */
CGLM_INLINE
void
glm_scale_uni(mat4 m, float s) {
  CGLM_ALIGN(8) vec3 v = { s, s, s };
  glm_scale_to(m, v, m);
}

/*!
 * @brief rotate existing transform matrix around X axis by angle
 *        and store result in dest
 *
 * @param[in]   m      affine transfrom
 * @param[in]   angle  angle (radians)
 * @param[out]  dest   rotated matrix
 */
CGLM_INLINE
void
glm_rotate_x(mat4 m, float angle, mat4 dest) {
  CGLM_ALIGN_MAT mat4 t = GLM_MAT4_IDENTITY_INIT;
  float c, s;

  c = cosf(angle);
  s = sinf(angle);

  t[1][1] =  c;
  t[1][2] =  s;
  t[2][1] = -s;
  t[2][2] =  c;

  glm_mul_rot(m, t, dest);
}

/*!
 * @brief rotate existing transform matrix around Y axis by angle
 *        and store result in dest
 *
 * @param[in]   m      affine transfrom
 * @param[in]   angle  angle (radians)
 * @param[out]  dest   rotated matrix
 */
CGLM_INLINE
void
glm_rotate_y(mat4 m, float angle, mat4 dest) {
  CGLM_ALIGN_MAT mat4 t = GLM_MAT4_IDENTITY_INIT;
  float c, s;

  c = cosf(angle);
  s = sinf(angle);

  t[0][0] =  c;
  t[0][2] = -s;
  t[2][0] =  s;
  t[2][2] =  c;

  glm_mul_rot(m, t, dest);
}

/*!
 * @brief rotate existing transform matrix around Z axis by angle
 *        and store result in dest
 *
 * @param[in]   m      affine transfrom
 * @param[in]   angle  angle (radians)
 * @param[out]  dest   rotated matrix
 */
CGLM_INLINE
void
glm_rotate_z(mat4 m, float angle, mat4 dest) {
  CGLM_ALIGN_MAT mat4 t = GLM_MAT4_IDENTITY_INIT;
  float c, s;

  c = cosf(angle);
  s = sinf(angle);

  t[0][0] =  c;
  t[0][1] =  s;
  t[1][0] = -s;
  t[1][1] =  c;

  glm_mul_rot(m, t, dest);
}

/*!
 * @brief creates NEW rotation matrix by angle and axis
 *
 * axis will be normalized so you don't need to normalize it
 *
 * @param[out] m     affine transfrom
 * @param[in]  angle angle (radians)
 * @param[in]  axis  axis
 */
CGLM_INLINE
void
glm_rotate_make(mat4 m, float angle, vec3 axis) {
  CGLM_ALIGN(8) vec3 axisn, v, vs;
  float c;

  c = cosf(angle);

  glm_vec3_normalize_to(axis, axisn);
  glm_vec3_scale(axisn, 1.0f - c, v);
  glm_vec3_scale(axisn, sinf(angle), vs);

  glm_vec3_scale(axisn, v[0], m[0]);
  glm_vec3_scale(axisn, v[1], m[1]);
  glm_vec3_scale(axisn, v[2], m[2]);

  m[0][0] += c;       m[1][0] -= vs[2];   m[2][0] += vs[1];
  m[0][1] += vs[2];   m[1][1] += c;       m[2][1] -= vs[0];
  m[0][2] -= vs[1];   m[1][2] += vs[0];   m[2][2] += c;

  m[0][3] = m[1][3] = m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0.0f;
  m[3][3] = 1.0f;
}

/*!
 * @brief rotate existing transform matrix around given axis by angle
 *
 * @param[in, out]  m      affine transfrom
 * @param[in]       angle  angle (radians)
 * @param[in]       axis   axis
 */
CGLM_INLINE
void
glm_rotate(mat4 m, float angle, vec3 axis) {
  CGLM_ALIGN_MAT mat4 rot;
  glm_rotate_make(rot, angle, axis);
  glm_mul_rot(m, rot, m);
}

/*!
 * @brief rotate existing transform
 *        around given axis by angle at given pivot point (rotation center)
 *
 * @param[in, out]  m      affine transfrom
 * @param[in]       pivot  rotation center
 * @param[in]       angle  angle (radians)
 * @param[in]       axis   axis
 */
CGLM_INLINE
void
glm_rotate_at(mat4 m, vec3 pivot, float angle, vec3 axis) {
  CGLM_ALIGN(8) vec3 pivotInv;

  glm_vec3_negate_to(pivot, pivotInv);

  glm_translate(m, pivot);
  glm_rotate(m, angle, axis);
  glm_translate(m, pivotInv);
}

/*!
 * @brief creates NEW rotation matrix by angle and axis at given point
 *
 * this creates rotation matrix, it assumes you don't have a matrix
 *
 * this should work faster than glm_rotate_at because it reduces
 * one glm_translate.
 *
 * @param[out] m      affine transfrom
 * @param[in]  pivot  rotation center
 * @param[in]  angle  angle (radians)
 * @param[in]  axis   axis
 */
CGLM_INLINE
void
glm_rotate_atm(mat4 m, vec3 pivot, float angle, vec3 axis) {
  CGLM_ALIGN(8) vec3 pivotInv;

  glm_vec3_negate_to(pivot, pivotInv);

  glm_translate_make(m, pivot);
  glm_rotate(m, angle, axis);
  glm_translate(m, pivotInv);
}

/*!
 * @brief decompose scale vector
 *
 * @param[in]  m  affine transform
 * @param[out] s  scale vector (Sx, Sy, Sz)
 */
CGLM_INLINE
void
glm_decompose_scalev(mat4 m, vec3 s) {
  s[0] = glm_vec3_norm(m[0]);
  s[1] = glm_vec3_norm(m[1]);
  s[2] = glm_vec3_norm(m[2]);
}

/*!
 * @brief returns true if matrix is uniform scaled. This is helpful for
 *        creating normal matrix.
 *
 * @param[in] m m
 *
 * @return boolean
 */
CGLM_INLINE
bool
glm_uniscaled(mat4 m) {
  CGLM_ALIGN(8) vec3 s;
  glm_decompose_scalev(m, s);
  return glm_vec3_eq_all(s);
}

/*!
 * @brief decompose rotation matrix (mat4) and scale vector [Sx, Sy, Sz]
 *        DON'T pass projected matrix here
 *
 * @param[in]  m affine transform
 * @param[out] r rotation matrix
 * @param[out] s scale matrix
 */
CGLM_INLINE
void
glm_decompose_rs(mat4 m, mat4 r, vec3 s) {
  CGLM_ALIGN(16) vec4 t = {0.0f, 0.0f, 0.0f, 1.0f};
  CGLM_ALIGN(8)  vec3 v;

  glm_vec4_copy(m[0], r[0]);
  glm_vec4_copy(m[1], r[1]);
  glm_vec4_copy(m[2], r[2]);
  glm_vec4_copy(t,    r[3]);

  s[0] = glm_vec3_norm(m[0]);
  s[1] = glm_vec3_norm(m[1]);
  s[2] = glm_vec3_norm(m[2]);

  glm_vec4_scale(r[0], 1.0f/s[0], r[0]);
  glm_vec4_scale(r[1], 1.0f/s[1], r[1]);
  glm_vec4_scale(r[2], 1.0f/s[2], r[2]);

  /* Note from Apple Open Source (asume that the matrix is orthonormal):
     check for a coordinate system flip.  If the determinant
     is -1, then negate the matrix and the scaling factors. */
  glm_vec3_cross(m[0], m[1], v);
  if (glm_vec3_dot(v, m[2]) < 0.0f) {
    glm_vec4_negate(r[0]);
    glm_vec4_negate(r[1]);
    glm_vec4_negate(r[2]);
    glm_vec3_negate(s);
  }
}

/*!
 * @brief decompose affine transform, TODO: extract shear factors.
 *        DON'T pass projected matrix here
 *
 * @param[in]  m affine transfrom
 * @param[out] t translation vector
 * @param[out] r rotation matrix (mat4)
 * @param[out] s scaling vector [X, Y, Z]
 */
CGLM_INLINE
void
glm_decompose(mat4 m, vec4 t, mat4 r, vec3 s) {
  glm_vec4_copy(m[3], t);
  glm_decompose_rs(m, r, s);
}

#endif /* cglm_affine_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*
 Functions:
   CGLM_INLINE void  glm_frustum(float left,    float right,
                                 float bottom,  float top,
                                 float nearVal, float farVal,
                                 mat4  dest)
   CGLM_INLINE void  glm_ortho(float left,    float right,
                               float bottom,  float top,
                               float nearVal, float farVal,
                               mat4  dest)
   CGLM_INLINE void  glm_ortho_aabb(vec3 box[2], mat4 dest)
   CGLM_INLINE void  glm_ortho_aabb_p(vec3 box[2],  float padding, mat4 dest)
   CGLM_INLINE void  glm_ortho_aabb_pz(vec3 box[2], float padding, mat4 dest)
   CGLM_INLINE void  glm_ortho_default(float aspect, mat4  dest)
   CGLM_INLINE void  glm_ortho_default_s(float aspect, float size, mat4 dest)
   CGLM_INLINE void  glm_perspective(float fovy,
                                     float aspect,
                                     float nearVal,
                                     float farVal,
                                     mat4  dest)
   CGLM_INLINE void  glm_perspective_default(float aspect, mat4 dest)
   CGLM_INLINE void  glm_perspective_resize(float aspect, mat4 proj)
   CGLM_INLINE void  glm_lookat(vec3 eye, vec3 center, vec3 up, mat4 dest)
   CGLM_INLINE void  glm_look(vec3 eye, vec3 dir, vec3 up, mat4 dest)
   CGLM_INLINE void  glm_look_anyup(vec3 eye, vec3 dir, mat4 dest)
   CGLM_INLINE void  glm_persp_decomp(mat4   proj,
                                      float *nearVal, float *farVal,
                                      float *top,     float *bottom,
                                      float *left,    float *right)
   CGLM_INLINE void  glm_persp_decompv(mat4 proj, float dest[6])
   CGLM_INLINE void  glm_persp_decomp_x(mat4 proj, float *left, float *right)
   CGLM_INLINE void  glm_persp_decomp_y(mat4 proj, float *top,  float *bottom)
   CGLM_INLINE void  glm_persp_decomp_z(mat4 proj, float *nearv, float *farv)
   CGLM_INLINE void  glm_persp_decomp_far(mat4 proj, float *farVal)
   CGLM_INLINE void  glm_persp_decomp_near(mat4 proj, float *nearVal)
   CGLM_INLINE float glm_persp_fovy(mat4 proj)
   CGLM_INLINE float glm_persp_aspect(mat4 proj)
   CGLM_INLINE void  glm_persp_sizes(mat4 proj, float fovy, vec4 dest)
 */

#ifndef cglm_vcam_h
#define cglm_vcam_h

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_plane_h
#define cglm_plane_h


/*
 Plane equation:  Ax + By + Cz + D = 0;

 It stored in vec4 as [A, B, C, D]. (A, B, C) is normal and D is distance
*/

/*
 Functions:
   CGLM_INLINE void  glm_plane_normalize(vec4 plane);
 */

/*!
 * @brief normalizes a plane
 *
 * @param[in, out] plane plane to normalize
 */
CGLM_INLINE
void
glm_plane_normalize(vec4 plane) {
  float norm;
  
  if ((norm = glm_vec3_norm(plane)) == 0.0f) {
    glm_vec4_zero(plane);
    return;
  }
  
  glm_vec4_scale(plane, 1.0f / norm, plane);
}

#endif /* cglm_plane_h */


/*!
 * @brief set up perspective peprojection matrix
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearVal near clipping plane
 * @param[in]  farVal  far clipping plane
 * @param[out] dest    result matrix
 */
CGLM_INLINE
void
glm_frustum(float left,    float right,
            float bottom,  float top,
            float nearVal, float farVal,
            mat4  dest) {
  float rl, tb, fn, nv;

  glm_mat4_zero(dest);

  rl = 1.0f / (right  - left);
  tb = 1.0f / (top    - bottom);
  fn =-1.0f / (farVal - nearVal);
  nv = 2.0f * nearVal;

  dest[0][0] = nv * rl;
  dest[1][1] = nv * tb;
  dest[2][0] = (right  + left)    * rl;
  dest[2][1] = (top    + bottom)  * tb;
  dest[2][2] = (farVal + nearVal) * fn;
  dest[2][3] =-1.0f;
  dest[3][2] = farVal * nv * fn;
}

/*!
 * @brief set up orthographic projection matrix
 *
 * @param[in]  left    viewport.left
 * @param[in]  right   viewport.right
 * @param[in]  bottom  viewport.bottom
 * @param[in]  top     viewport.top
 * @param[in]  nearVal near clipping plane
 * @param[in]  farVal  far clipping plane
 * @param[out] dest    result matrix
 */
CGLM_INLINE
void
glm_ortho(float left,    float right,
          float bottom,  float top,
          float nearVal, float farVal,
          mat4  dest) {
  float rl, tb, fn;

  glm_mat4_zero(dest);

  rl = 1.0f / (right  - left);
  tb = 1.0f / (top    - bottom);
  fn =-1.0f / (farVal - nearVal);

  dest[0][0] = 2.0f * rl;
  dest[1][1] = 2.0f * tb;
  dest[2][2] = 2.0f * fn;
  dest[3][0] =-(right  + left)    * rl;
  dest[3][1] =-(top    + bottom)  * tb;
  dest[3][2] = (farVal + nearVal) * fn;
  dest[3][3] = 1.0f;
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box   AABB
 * @param[out] dest  result matrix
 */
CGLM_INLINE
void
glm_ortho_aabb(vec3 box[2], mat4 dest) {
  glm_ortho(box[0][0],  box[1][0],
            box[0][1],  box[1][1],
           -box[1][2], -box[0][2],
            dest);
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding
 * @param[out] dest    result matrix
 */
CGLM_INLINE
void
glm_ortho_aabb_p(vec3 box[2], float padding, mat4 dest) {
  glm_ortho(box[0][0] - padding,    box[1][0] + padding,
            box[0][1] - padding,    box[1][1] + padding,
          -(box[1][2] + padding), -(box[0][2] - padding),
            dest);
}

/*!
 * @brief set up orthographic projection matrix using bounding box
 *
 * bounding box (AABB) must be in view space
 *
 * @param[in]  box     AABB
 * @param[in]  padding padding for near and far
 * @param[out] dest    result matrix
 */
CGLM_INLINE
void
glm_ortho_aabb_pz(vec3 box[2], float padding, mat4 dest) {
  glm_ortho(box[0][0],              box[1][0],
            box[0][1],              box[1][1],
          -(box[1][2] + padding), -(box[0][2] - padding),
            dest);
}

/*!
 * @brief set up unit orthographic projection matrix
 *
 * @param[in]  aspect aspect ration ( width / height )
 * @param[out] dest   result matrix
 */
CGLM_INLINE
void
glm_ortho_default(float aspect, mat4 dest) {
  if (aspect >= 1.0f) {
    glm_ortho(-aspect, aspect, -1.0f, 1.0f, -100.0f, 100.0f, dest);
    return;
  }

  aspect = 1.0f / aspect;

  glm_ortho(-1.0f, 1.0f, -aspect, aspect, -100.0f, 100.0f, dest);
}

/*!
 * @brief set up orthographic projection matrix with given CUBE size
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[in]  size   cube size
 * @param[out] dest   result matrix
 */
CGLM_INLINE
void
glm_ortho_default_s(float aspect, float size, mat4 dest) {
  if (aspect >= 1.0f) {
    glm_ortho(-size * aspect,
               size * aspect,
              -size,
               size,
              -size - 100.0f,
               size + 100.0f,
               dest);
    return;
  }

  glm_ortho(-size,
             size,
            -size / aspect,
             size / aspect,
            -size - 100.0f,
             size + 100.0f,
             dest);
}

/*!
 * @brief set up perspective projection matrix
 *
 * @param[in]  fovy    field of view angle
 * @param[in]  aspect  aspect ratio ( width / height )
 * @param[in]  nearVal near clipping plane
 * @param[in]  farVal  far clipping planes
 * @param[out] dest    result matrix
 */
CGLM_INLINE
void
glm_perspective(float fovy,
                float aspect,
                float nearVal,
                float farVal,
                mat4  dest) {
  float f, fn;

  glm_mat4_zero(dest);

  f  = 1.0f / tanf(fovy * 0.5f);
  fn = 1.0f / (nearVal - farVal);

  dest[0][0] = f / aspect;
  dest[1][1] = f;
  dest[2][2] = (nearVal + farVal) * fn;
  dest[2][3] =-1.0f;
  dest[3][2] = 2.0f * nearVal * farVal * fn;
}

/*!
 * @brief extend perspective projection matrix's far distance
 *
 * this function does not guarantee far >= near, be aware of that!
 *
 * @param[in, out] proj      projection matrix to extend
 * @param[in]      deltaFar  distance from existing far (negative to shink)
 */
CGLM_INLINE
void
glm_persp_move_far(mat4 proj, float deltaFar) {
  float fn, farVal, nearVal, p22, p32;

  p22        = proj[2][2];
  p32        = proj[3][2];

  nearVal    = p32 / (p22 - 1.0f);
  farVal     = p32 / (p22 + 1.0f) + deltaFar;
  fn         = 1.0f / (nearVal - farVal);

  proj[2][2] = (nearVal + farVal) * fn;
  proj[3][2] = 2.0f * nearVal * farVal * fn;
}

/*!
 * @brief set up perspective projection matrix with default near/far
 *        and angle values
 *
 * @param[in]  aspect aspect ratio ( width / height )
 * @param[out] dest   result matrix
 */
CGLM_INLINE
void
glm_perspective_default(float aspect, mat4 dest) {
  glm_perspective(GLM_PI_4f, aspect, 0.01f, 100.0f, dest);
}

/*!
 * @brief resize perspective matrix by aspect ratio ( width / height )
 *        this makes very easy to resize proj matrix when window /viewport
 *        reized
 *
 * @param[in]      aspect aspect ratio ( width / height )
 * @param[in, out] proj   perspective projection matrix
 */
CGLM_INLINE
void
glm_perspective_resize(float aspect, mat4 proj) {
  if (proj[0][0] == 0.0f)
    return;

  proj[0][0] = proj[1][1] / aspect;
}

/*!
 * @brief set up view matrix
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  center center vector
 * @param[in]  up     up vector
 * @param[out] dest   result matrix
 */
CGLM_INLINE
void
glm_lookat(vec3 eye, vec3 center, vec3 up, mat4 dest) {
  CGLM_ALIGN(8) vec3 f, u, s;

  glm_vec3_sub(center, eye, f);
  glm_vec3_normalize(f);

  glm_vec3_crossn(f, up, s);
  glm_vec3_cross(s, f, u);

  dest[0][0] = s[0];
  dest[0][1] = u[0];
  dest[0][2] =-f[0];
  dest[1][0] = s[1];
  dest[1][1] = u[1];
  dest[1][2] =-f[1];
  dest[2][0] = s[2];
  dest[2][1] = u[2];
  dest[2][2] =-f[2];
  dest[3][0] =-glm_vec3_dot(s, eye);
  dest[3][1] =-glm_vec3_dot(u, eye);
  dest[3][2] = glm_vec3_dot(f, eye);
  dest[0][3] = dest[1][3] = dest[2][3] = 0.0f;
  dest[3][3] = 1.0f;
}

/*!
 * @brief set up view matrix
 *
 * convenient wrapper for lookat: if you only have direction not target self
 * then this might be useful. Because you need to get target from direction.
 *
 * NOTE: The UP vector must not be parallel to the line of sight from
 *       the eye point to the reference point
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[in]  up     up vector
 * @param[out] dest   result matrix
 */
CGLM_INLINE
void
glm_look(vec3 eye, vec3 dir, vec3 up, mat4 dest) {
  CGLM_ALIGN(8) vec3 target;
  glm_vec3_add(eye, dir, target);
  glm_lookat(eye, target, up, dest);
}

/*!
 * @brief set up view matrix
 *
 * convenient wrapper for look: if you only have direction and if you don't
 * care what UP vector is then this might be useful to create view matrix
 *
 * @param[in]  eye    eye vector
 * @param[in]  dir    direction vector
 * @param[out] dest   result matrix
 */
CGLM_INLINE
void
glm_look_anyup(vec3 eye, vec3 dir, mat4 dest) {
  CGLM_ALIGN(8) vec3 up;
  glm_vec3_ortho(dir, up);
  glm_look(eye, dir, up, dest);
}

/*!
 * @brief decomposes frustum values of perspective projection.
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearVal near
 * @param[out] farVal  far
 * @param[out] top     top
 * @param[out] bottom  bottom
 * @param[out] left    left
 * @param[out] right   right
 */
CGLM_INLINE
void
glm_persp_decomp(mat4 proj,
                 float * __restrict nearVal, float * __restrict farVal,
                 float * __restrict top,     float * __restrict bottom,
                 float * __restrict left,    float * __restrict right) {
  float m00, m11, m20, m21, m22, m32, n, f;
  float n_m11, n_m00;

  m00 = proj[0][0];
  m11 = proj[1][1];
  m20 = proj[2][0];
  m21 = proj[2][1];
  m22 = proj[2][2];
  m32 = proj[3][2];

  n = m32 / (m22 - 1.0f);
  f = m32 / (m22 + 1.0f);

  n_m11 = n / m11;
  n_m00 = n / m00;

  *nearVal = n;
  *farVal  = f;
  *bottom  = n_m11 * (m21 - 1.0f);
  *top     = n_m11 * (m21 + 1.0f);
  *left    = n_m00 * (m20 - 1.0f);
  *right   = n_m00 * (m20 + 1.0f);
}

/*!
 * @brief decomposes frustum values of perspective projection.
 *        this makes easy to get all values at once
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] dest   array
 */
CGLM_INLINE
void
glm_persp_decompv(mat4 proj, float dest[6]) {
  glm_persp_decomp(proj, &dest[0], &dest[1], &dest[2],
                         &dest[3], &dest[4], &dest[5]);
}

/*!
 * @brief decomposes left and right values of perspective projection.
 *        x stands for x axis (left / right axis)
 *
 * @param[in]  proj  perspective projection matrix
 * @param[out] left  left
 * @param[out] right right
 */
CGLM_INLINE
void
glm_persp_decomp_x(mat4 proj,
                   float * __restrict left,
                   float * __restrict right) {
  float nearVal, m20, m00;

  m00 = proj[0][0];
  m20 = proj[2][0];

  nearVal = proj[3][2] / (proj[3][3] - 1.0f);
  *left   = nearVal * (m20 - 1.0f) / m00;
  *right  = nearVal * (m20 + 1.0f) / m00;
}

/*!
 * @brief decomposes top and bottom values of perspective projection.
 *        y stands for y axis (top / botom axis)
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] top    top
 * @param[out] bottom bottom
 */
CGLM_INLINE
void
glm_persp_decomp_y(mat4 proj,
                   float * __restrict top,
                   float * __restrict bottom) {
  float nearVal, m21, m11;

  m21 = proj[2][1];
  m11 = proj[1][1];

  nearVal = proj[3][2] / (proj[3][3] - 1.0f);
  *bottom = nearVal * (m21 - 1) / m11;
  *top    = nearVal * (m21 + 1) / m11;
}

/*!
 * @brief decomposes near and far values of perspective projection.
 *        z stands for z axis (near / far axis)
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearVal near
 * @param[out] farVal  far
 */
CGLM_INLINE
void
glm_persp_decomp_z(mat4 proj,
                   float * __restrict nearVal,
                   float * __restrict farVal) {
  float m32, m22;

  m32 = proj[3][2];
  m22 = proj[2][2];

  *nearVal = m32 / (m22 - 1.0f);
  *farVal  = m32 / (m22 + 1.0f);
}

/*!
 * @brief decomposes far value of perspective projection.
 *
 * @param[in]  proj   perspective projection matrix
 * @param[out] farVal far
 */
CGLM_INLINE
void
glm_persp_decomp_far(mat4 proj, float * __restrict farVal) {
  *farVal = proj[3][2] / (proj[2][2] + 1.0f);
}

/*!
 * @brief decomposes near value of perspective projection.
 *
 * @param[in]  proj    perspective projection matrix
 * @param[out] nearVal near
 */
CGLM_INLINE
void
glm_persp_decomp_near(mat4 proj, float * __restrict nearVal) {
  *nearVal = proj[3][2] / (proj[2][2] - 1.0f);
}

/*!
 * @brief returns field of view angle along the Y-axis (in radians)
 *
 * if you need to degrees, use glm_deg to convert it or use this:
 * fovy_deg = glm_deg(glm_persp_fovy(projMatrix))
 *
 * @param[in] proj perspective projection matrix
 */
CGLM_INLINE
float
glm_persp_fovy(mat4 proj) {
  return 2.0f * atanf(1.0f / proj[1][1]);
}

/*!
 * @brief returns aspect ratio of perspective projection
 *
 * @param[in] proj perspective projection matrix
 */
CGLM_INLINE
float
glm_persp_aspect(mat4 proj) {
  return proj[1][1] / proj[0][0];
}

/*!
 * @brief returns sizes of near and far planes of perspective projection
 *
 * @param[in]  proj perspective projection matrix
 * @param[in]  fovy fovy (see brief)
 * @param[out] dest sizes order: [Wnear, Hnear, Wfar, Hfar]
 */
CGLM_INLINE
void
glm_persp_sizes(mat4 proj, float fovy, vec4 dest) {
  float t, a, nearVal, farVal;

  t = 2.0f * tanf(fovy * 0.5f);
  a = glm_persp_aspect(proj);

  glm_persp_decomp_z(proj, &nearVal, &farVal);

  dest[1]  = t * nearVal;
  dest[3]  = t * farVal;
  dest[0]  = a * dest[1];
  dest[2]  = a * dest[3];
}

#endif /* cglm_vcam_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_frustum_h
#define cglm_frustum_h


#define GLM_LBN 0 /* left  bottom near */
#define GLM_LTN 1 /* left  top    near */
#define GLM_RTN 2 /* right top    near */
#define GLM_RBN 3 /* right bottom near */

#define GLM_LBF 4 /* left  bottom far  */
#define GLM_LTF 5 /* left  top    far  */
#define GLM_RTF 6 /* right top    far  */
#define GLM_RBF 7 /* right bottom far  */

#define GLM_LEFT   0
#define GLM_RIGHT  1
#define GLM_BOTTOM 2
#define GLM_TOP    3
#define GLM_NEAR   4
#define GLM_FAR    5

/* you can override clip space coords
   but you have to provide all with same name
   e.g.: define GLM_CSCOORD_LBN {0.0f, 0.0f, 1.0f, 1.0f} */
#ifndef GLM_CUSTOM_CLIPSPACE

/* near */
#define GLM_CSCOORD_LBN {-1.0f, -1.0f, -1.0f, 1.0f}
#define GLM_CSCOORD_LTN {-1.0f,  1.0f, -1.0f, 1.0f}
#define GLM_CSCOORD_RTN { 1.0f,  1.0f, -1.0f, 1.0f}
#define GLM_CSCOORD_RBN { 1.0f, -1.0f, -1.0f, 1.0f}

/* far */
#define GLM_CSCOORD_LBF {-1.0f, -1.0f,  1.0f, 1.0f}
#define GLM_CSCOORD_LTF {-1.0f,  1.0f,  1.0f, 1.0f}
#define GLM_CSCOORD_RTF { 1.0f,  1.0f,  1.0f, 1.0f}
#define GLM_CSCOORD_RBF { 1.0f, -1.0f,  1.0f, 1.0f}

#endif

/*!
 * @brief extracts view frustum planes
 *
 * planes' space:
 *  1- if m = proj:     View Space
 *  2- if m = viewProj: World Space
 *  3- if m = MVP:      Object Space
 *
 * You probably want to extract planes in world space so use viewProj as m
 * Computing viewProj:
 *   glm_mat4_mul(proj, view, viewProj);
 *
 * Exracted planes order: [left, right, bottom, top, near, far]
 *
 * @param[in]  m    matrix (see brief)
 * @param[out] dest extracted view frustum planes (see brief)
 */
CGLM_INLINE
void
glm_frustum_planes(mat4 m, vec4 dest[6]) {
  mat4 t;

  glm_mat4_transpose_to(m, t);

  glm_vec4_add(t[3], t[0], dest[0]); /* left   */
  glm_vec4_sub(t[3], t[0], dest[1]); /* right  */
  glm_vec4_add(t[3], t[1], dest[2]); /* bottom */
  glm_vec4_sub(t[3], t[1], dest[3]); /* top    */
  glm_vec4_add(t[3], t[2], dest[4]); /* near   */
  glm_vec4_sub(t[3], t[2], dest[5]); /* far    */

  glm_plane_normalize(dest[0]);
  glm_plane_normalize(dest[1]);
  glm_plane_normalize(dest[2]);
  glm_plane_normalize(dest[3]);
  glm_plane_normalize(dest[4]);
  glm_plane_normalize(dest[5]);
}

/*!
 * @brief extracts view frustum corners using clip-space coordinates
 *
 * corners' space:
 *  1- if m = invViewProj: World Space
 *  2- if m = invMVP:      Object Space
 *
 * You probably want to extract corners in world space so use invViewProj
 * Computing invViewProj:
 *   glm_mat4_mul(proj, view, viewProj);
 *   ...
 *   glm_mat4_inv(viewProj, invViewProj);
 *
 * if you have a near coord at i index, you can get it's far coord by i + 4
 *
 * Find center coordinates:
 *   for (j = 0; j < 4; j++) {
 *     glm_vec3_center(corners[i], corners[i + 4], centerCorners[i]);
 *   }
 *
 * @param[in]  invMat matrix (see brief)
 * @param[out] dest   exracted view frustum corners (see brief)
 */
CGLM_INLINE
void
glm_frustum_corners(mat4 invMat, vec4 dest[8]) {
  vec4 c[8];

  /* indexOf(nearCoord) = indexOf(farCoord) + 4 */
  vec4 csCoords[8] = {
    GLM_CSCOORD_LBN,
    GLM_CSCOORD_LTN,
    GLM_CSCOORD_RTN,
    GLM_CSCOORD_RBN,

    GLM_CSCOORD_LBF,
    GLM_CSCOORD_LTF,
    GLM_CSCOORD_RTF,
    GLM_CSCOORD_RBF
  };

  glm_mat4_mulv(invMat, csCoords[0], c[0]);
  glm_mat4_mulv(invMat, csCoords[1], c[1]);
  glm_mat4_mulv(invMat, csCoords[2], c[2]);
  glm_mat4_mulv(invMat, csCoords[3], c[3]);
  glm_mat4_mulv(invMat, csCoords[4], c[4]);
  glm_mat4_mulv(invMat, csCoords[5], c[5]);
  glm_mat4_mulv(invMat, csCoords[6], c[6]);
  glm_mat4_mulv(invMat, csCoords[7], c[7]);

  glm_vec4_scale(c[0], 1.0f / c[0][3], dest[0]);
  glm_vec4_scale(c[1], 1.0f / c[1][3], dest[1]);
  glm_vec4_scale(c[2], 1.0f / c[2][3], dest[2]);
  glm_vec4_scale(c[3], 1.0f / c[3][3], dest[3]);
  glm_vec4_scale(c[4], 1.0f / c[4][3], dest[4]);
  glm_vec4_scale(c[5], 1.0f / c[5][3], dest[5]);
  glm_vec4_scale(c[6], 1.0f / c[6][3], dest[6]);
  glm_vec4_scale(c[7], 1.0f / c[7][3], dest[7]);
}

/*!
 * @brief finds center of view frustum
 *
 * @param[in]  corners view frustum corners
 * @param[out] dest    view frustum center
 */
CGLM_INLINE
void
glm_frustum_center(vec4 corners[8], vec4 dest) {
  vec4 center;

  glm_vec4_copy(corners[0], center);

  glm_vec4_add(corners[1], center, center);
  glm_vec4_add(corners[2], center, center);
  glm_vec4_add(corners[3], center, center);
  glm_vec4_add(corners[4], center, center);
  glm_vec4_add(corners[5], center, center);
  glm_vec4_add(corners[6], center, center);
  glm_vec4_add(corners[7], center, center);

  glm_vec4_scale(center, 0.125f, dest);
}

/*!
 * @brief finds bounding box of frustum relative to given matrix e.g. view mat
 *
 * @param[in]  corners view frustum corners
 * @param[in]  m       matrix to convert existing conners
 * @param[out] box     bounding box as array [min, max]
 */
CGLM_INLINE
void
glm_frustum_box(vec4 corners[8], mat4 m, vec3 box[2]) {
  vec4 v;
  vec3 min, max;
  int  i;

  glm_vec3_broadcast(FLT_MAX, min);
  glm_vec3_broadcast(-FLT_MAX, max);

  for (i = 0; i < 8; i++) {
    glm_mat4_mulv(m, corners[i], v);

    min[0] = glm_min(min[0], v[0]);
    min[1] = glm_min(min[1], v[1]);
    min[2] = glm_min(min[2], v[2]);

    max[0] = glm_max(max[0], v[0]);
    max[1] = glm_max(max[1], v[1]);
    max[2] = glm_max(max[2], v[2]);
  }

  glm_vec3_copy(min, box[0]);
  glm_vec3_copy(max, box[1]);
}

/*!
 * @brief finds planes corners which is between near and far planes (parallel)
 *
 * this will be helpful if you want to split a frustum e.g. CSM/PSSM. This will
 * find planes' corners but you will need to one more plane.
 * Actually you have it, it is near, far or created previously with this func ;)
 *
 * @param[in]  corners view  frustum corners
 * @param[in]  splitDist     split distance
 * @param[in]  farDist       far distance (zFar)
 * @param[out] planeCorners  plane corners [LB, LT, RT, RB]
 */
CGLM_INLINE
void
glm_frustum_corners_at(vec4  corners[8],
                       float splitDist,
                       float farDist,
                       vec4  planeCorners[4]) {
  vec4  corner;
  float dist, sc;

  /* because distance and scale is same for all */
  dist = glm_vec3_distance(corners[GLM_RTF], corners[GLM_RTN]);
  sc   = dist * (splitDist / farDist);

  /* left bottom */
  glm_vec4_sub(corners[GLM_LBF], corners[GLM_LBN], corner);
  glm_vec4_scale_as(corner, sc, corner);
  glm_vec4_add(corners[GLM_LBN], corner, planeCorners[0]);

  /* left top */
  glm_vec4_sub(corners[GLM_LTF], corners[GLM_LTN], corner);
  glm_vec4_scale_as(corner, sc, corner);
  glm_vec4_add(corners[GLM_LTN], corner, planeCorners[1]);

  /* right top */
  glm_vec4_sub(corners[GLM_RTF], corners[GLM_RTN], corner);
  glm_vec4_scale_as(corner, sc, corner);
  glm_vec4_add(corners[GLM_RTN], corner, planeCorners[2]);

  /* right bottom */
  glm_vec4_sub(corners[GLM_RBF], corners[GLM_RBN], corner);
  glm_vec4_scale_as(corner, sc, corner);
  glm_vec4_add(corners[GLM_RBN], corner, planeCorners[3]);
}

#endif /* cglm_frustum_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*
 Macros:
   GLM_QUAT_IDENTITY_INIT
   GLM_QUAT_IDENTITY

 Functions:
   CGLM_INLINE void glm_quat_identity(versor q);
   CGLM_INLINE void glm_quat_init(versor q, float x, float y, float z, float w);
   CGLM_INLINE void glm_quat(versor q, float angle, float x, float y, float z);
   CGLM_INLINE void glm_quatv(versor q, float angle, vec3 axis);
   CGLM_INLINE void glm_quat_copy(versor q, versor dest);
   CGLM_INLINE float glm_quat_norm(versor q);
   CGLM_INLINE void glm_quat_normalize(versor q);
   CGLM_INLINE void glm_quat_normalize_to(versor q, versor dest);
   CGLM_INLINE float glm_quat_dot(versor p, versor q);
   CGLM_INLINE void glm_quat_conjugate(versor q, versor dest);
   CGLM_INLINE void glm_quat_inv(versor q, versor dest);
   CGLM_INLINE void glm_quat_add(versor p, versor q, versor dest);
   CGLM_INLINE void glm_quat_sub(versor p, versor q, versor dest);
   CGLM_INLINE float glm_quat_real(versor q);
   CGLM_INLINE void glm_quat_imag(versor q, vec3 dest);
   CGLM_INLINE void glm_quat_imagn(versor q, vec3 dest);
   CGLM_INLINE float glm_quat_imaglen(versor q);
   CGLM_INLINE float glm_quat_angle(versor q);
   CGLM_INLINE void glm_quat_axis(versor q, vec3 dest);
   CGLM_INLINE void glm_quat_mul(versor p, versor q, versor dest);
   CGLM_INLINE void glm_quat_mat4(versor q, mat4 dest);
   CGLM_INLINE void glm_quat_mat4t(versor q, mat4 dest);
   CGLM_INLINE void glm_quat_mat3(versor q, mat3 dest);
   CGLM_INLINE void glm_quat_mat3t(versor q, mat3 dest);
   CGLM_INLINE void glm_quat_lerp(versor from, versor to, float t, versor dest);
   CGLM_INLINE void glm_quat_lerpc(versor from, versor to, float t, versor dest);
   CGLM_INLINE void glm_quat_slerp(versor q, versor r, float t, versor dest);
   CGLM_INLINE void glm_quat_look(vec3 eye, versor ori, mat4 dest);
   CGLM_INLINE void glm_quat_for(vec3 dir, vec3 fwd, vec3 up, versor dest);
   CGLM_INLINE void glm_quat_forp(vec3 from,
                                  vec3 to,
                                  vec3 fwd,
                                  vec3 up,
                                  versor dest);
   CGLM_INLINE void glm_quat_rotatev(versor q, vec3 v, vec3 dest);
   CGLM_INLINE void glm_quat_rotate(mat4 m, versor q, mat4 dest);
 */

#ifndef cglm_quat_h
#define cglm_quat_h


#ifdef CGLM_SSE_FP
/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_quat_simd_h
#define cglm_quat_simd_h
#if defined( __SSE__ ) || defined( __SSE2__ )


CGLM_INLINE
void
glm_quat_mul_sse2(versor p, versor q, versor dest) {
  /*
   + (a1 b2 + b1 a2 + c1 d2 − d1 c2)i
   + (a1 c2 − b1 d2 + c1 a2 + d1 b2)j
   + (a1 d2 + b1 c2 − c1 b2 + d1 a2)k
     a1 a2 − b1 b2 − c1 c2 − d1 d2
   */

  __m128 xp, xq, x0, r;

  xp = glmm_load(p); /* 3 2 1 0 */
  xq = glmm_load(q);

  r  = _mm_mul_ps(glmm_shuff1x(xp, 3), xq);

  x0 = _mm_xor_ps(glmm_shuff1x(xp, 0), _mm_set_ps(-0.f, 0.f, -0.f, 0.f));
  r  = _mm_add_ps(r, _mm_mul_ps(x0, glmm_shuff1(xq, 0, 1, 2, 3)));

  x0 = _mm_xor_ps(glmm_shuff1x(xp, 1), _mm_set_ps(-0.f, -0.f, 0.f, 0.f));
  r  = _mm_add_ps(r, _mm_mul_ps(x0, glmm_shuff1(xq, 1, 0, 3, 2)));

  x0 = _mm_xor_ps(glmm_shuff1x(xp, 2), _mm_set_ps(-0.f, 0.f, 0.f, -0.f));
  r  = _mm_add_ps(r, _mm_mul_ps(x0, glmm_shuff1(xq, 2, 3, 0, 1)));

  glmm_store(dest, r);
}


#endif
#endif /* cglm_quat_simd_h */

#endif

CGLM_INLINE
void
glm_mat4_mulv(mat4 m, vec4 v, vec4 dest);

CGLM_INLINE
void
glm_mul_rot(mat4 m1, mat4 m2, mat4 dest);

CGLM_INLINE
void
glm_translate(mat4 m, vec3 v);

/*
 * IMPORTANT:
 * ----------------------------------------------------------------------------
 * cglm stores quat as [x, y, z, w] since v0.3.6
 *
 * it was [w, x, y, z] before v0.3.6 it has been changed to [x, y, z, w]
 * with v0.3.6 version.
 * ----------------------------------------------------------------------------
 */

#define GLM_QUAT_IDENTITY_INIT  {0.0f, 0.0f, 0.0f, 1.0f}
#define GLM_QUAT_IDENTITY       ((versor)GLM_QUAT_IDENTITY_INIT)

/*!
 * @brief makes given quat to identity
 *
 * @param[in, out]  q  quaternion
 */
CGLM_INLINE
void
glm_quat_identity(versor q) {
  CGLM_ALIGN(16) versor v = GLM_QUAT_IDENTITY_INIT;
  glm_vec4_copy(v, q);
}

/*!
 * @brief make given quaternion array's each element identity quaternion
 *
 * @param[in, out]  q     quat array (must be aligned (16)
 *                        if alignment is not disabled)
 *
 * @param[in]       count count of quaternions
 */
CGLM_INLINE
void
glm_quat_identity_array(versor * __restrict q, size_t count) {
  CGLM_ALIGN(16) versor v = GLM_QUAT_IDENTITY_INIT;
  size_t i;

  for (i = 0; i < count; i++) {
    glm_vec4_copy(v, q[i]);
  }
}

/*!
 * @brief inits quaterion with raw values
 *
 * @param[out]  q     quaternion
 * @param[in]   x     x
 * @param[in]   y     y
 * @param[in]   z     z
 * @param[in]   w     w (real part)
 */
CGLM_INLINE
void
glm_quat_init(versor q, float x, float y, float z, float w) {
  q[0] = x;
  q[1] = y;
  q[2] = z;
  q[3] = w;
}

/*!
 * @brief creates NEW quaternion with axis vector
 *
 * @param[out]  q     quaternion
 * @param[in]   angle angle (radians)
 * @param[in]   axis  axis
 */
CGLM_INLINE
void
glm_quatv(versor q, float angle, vec3 axis) {
  CGLM_ALIGN(8) vec3 k;
  float a, c, s;

  a = angle * 0.5f;
  c = cosf(a);
  s = sinf(a);

  glm_normalize_to(axis, k);

  q[0] = s * k[0];
  q[1] = s * k[1];
  q[2] = s * k[2];
  q[3] = c;
}

/*!
 * @brief creates NEW quaternion with individual axis components
 *
 * @param[out]  q     quaternion
 * @param[in]   angle angle (radians)
 * @param[in]   x     axis.x
 * @param[in]   y     axis.y
 * @param[in]   z     axis.z
 */
CGLM_INLINE
void
glm_quat(versor q, float angle, float x, float y, float z) {
  CGLM_ALIGN(8) vec3 axis = {x, y, z};
  glm_quatv(q, angle, axis);
}

/*!
 * @brief copy quaternion to another one
 *
 * @param[in]  q     quaternion
 * @param[out] dest  destination
 */
CGLM_INLINE
void
glm_quat_copy(versor q, versor dest) {
  glm_vec4_copy(q, dest);
}

/*!
 * @brief returns norm (magnitude) of quaternion
 *
 * @param[out]  q  quaternion
 */
CGLM_INLINE
float
glm_quat_norm(versor q) {
  return glm_vec4_norm(q);
}

/*!
 * @brief normalize quaternion and store result in dest
 *
 * @param[in]   q     quaternion to normalze
 * @param[out]  dest  destination quaternion
 */
CGLM_INLINE
void
glm_quat_normalize_to(versor q, versor dest) {
#if defined( __SSE2__ ) || defined( __SSE2__ )
  __m128 xdot, x0;
  float  dot;

  x0   = glmm_load(q);
  xdot = glmm_vdot(x0, x0);
  dot  = _mm_cvtss_f32(xdot);

  if (dot <= 0.0f) {
    glm_quat_identity(dest);
    return;
  }

  glmm_store(dest, _mm_div_ps(x0, _mm_sqrt_ps(xdot)));
#else
  float dot;

  dot = glm_vec4_norm2(q);

  if (dot <= 0.0f) {
    glm_quat_identity(dest);
    return;
  }

  glm_vec4_scale(q, 1.0f / sqrtf(dot), dest);
#endif
}

/*!
 * @brief normalize quaternion
 *
 * @param[in, out]  q  quaternion
 */
CGLM_INLINE
void
glm_quat_normalize(versor q) {
  glm_quat_normalize_to(q, q);
}

/*!
 * @brief dot product of two quaternion
 *
 * @param[in]  p  quaternion 1
 * @param[in]  q  quaternion 2
 */
CGLM_INLINE
float
glm_quat_dot(versor p, versor q) {
  return glm_vec4_dot(p, q);
}

/*!
 * @brief conjugate of quaternion
 *
 * @param[in]   q     quaternion
 * @param[out]  dest  conjugate
 */
CGLM_INLINE
void
glm_quat_conjugate(versor q, versor dest) {
  glm_vec4_negate_to(q, dest);
  dest[3] = -dest[3];
}

/*!
 * @brief inverse of non-zero quaternion
 *
 * @param[in]   q    quaternion
 * @param[out]  dest inverse quaternion
 */
CGLM_INLINE
void
glm_quat_inv(versor q, versor dest) {
  CGLM_ALIGN(16) versor conj;
  glm_quat_conjugate(q, conj);
  glm_vec4_scale(conj, 1.0f / glm_vec4_norm2(q), dest);
}

/*!
 * @brief add (componentwise) two quaternions and store result in dest
 *
 * @param[in]   p    quaternion 1
 * @param[in]   q    quaternion 2
 * @param[out]  dest result quaternion
 */
CGLM_INLINE
void
glm_quat_add(versor p, versor q, versor dest) {
  glm_vec4_add(p, q, dest);
}

/*!
 * @brief subtract (componentwise) two quaternions and store result in dest
 *
 * @param[in]   p    quaternion 1
 * @param[in]   q    quaternion 2
 * @param[out]  dest result quaternion
 */
CGLM_INLINE
void
glm_quat_sub(versor p, versor q, versor dest) {
  glm_vec4_sub(p, q, dest);
}

/*!
 * @brief returns real part of quaternion
 *
 * @param[in]   q    quaternion
 */
CGLM_INLINE
float
glm_quat_real(versor q) {
  return q[3];
}

/*!
 * @brief returns imaginary part of quaternion
 *
 * @param[in]   q    quaternion
 * @param[out]  dest imag
 */
CGLM_INLINE
void
glm_quat_imag(versor q, vec3 dest) {
  dest[0] = q[0];
  dest[1] = q[1];
  dest[2] = q[2];
}

/*!
 * @brief returns normalized imaginary part of quaternion
 *
 * @param[in]   q    quaternion
 */
CGLM_INLINE
void
glm_quat_imagn(versor q, vec3 dest) {
  glm_normalize_to(q, dest);
}

/*!
 * @brief returns length of imaginary part of quaternion
 *
 * @param[in]   q    quaternion
 */
CGLM_INLINE
float
glm_quat_imaglen(versor q) {
  return glm_vec3_norm(q);
}

/*!
 * @brief returns angle of quaternion
 *
 * @param[in]   q    quaternion
 */
CGLM_INLINE
float
glm_quat_angle(versor q) {
  /*
   sin(theta / 2) = length(x*x + y*y + z*z)
   cos(theta / 2) = w
   theta          = 2 * atan(sin(theta / 2) / cos(theta / 2))
   */
  return 2.0f * atan2f(glm_quat_imaglen(q), glm_quat_real(q));
}

/*!
 * @brief axis of quaternion
 *
 * @param[in]   q    quaternion
 * @param[out]  dest axis of quaternion
 */
CGLM_INLINE
void
glm_quat_axis(versor q, vec3 dest) {
  glm_quat_imagn(q, dest);
}

/*!
 * @brief multiplies two quaternion and stores result in dest
 *        this is also called Hamilton Product
 *
 * According to WikiPedia:
 * The product of two rotation quaternions [clarification needed] will be
 * equivalent to the rotation q followed by the rotation p
 *
 * @param[in]   p     quaternion 1
 * @param[in]   q     quaternion 2
 * @param[out]  dest  result quaternion
 */
CGLM_INLINE
void
glm_quat_mul(versor p, versor q, versor dest) {
  /*
    + (a1 b2 + b1 a2 + c1 d2 − d1 c2)i
    + (a1 c2 − b1 d2 + c1 a2 + d1 b2)j
    + (a1 d2 + b1 c2 − c1 b2 + d1 a2)k
       a1 a2 − b1 b2 − c1 c2 − d1 d2
   */
#if defined( __SSE__ ) || defined( __SSE2__ )
  glm_quat_mul_sse2(p, q, dest);
#else
  dest[0] = p[3] * q[0] + p[0] * q[3] + p[1] * q[2] - p[2] * q[1];
  dest[1] = p[3] * q[1] - p[0] * q[2] + p[1] * q[3] + p[2] * q[0];
  dest[2] = p[3] * q[2] + p[0] * q[1] - p[1] * q[0] + p[2] * q[3];
  dest[3] = p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2];
#endif
}

/*!
 * @brief convert quaternion to mat4
 *
 * @param[in]   q     quaternion
 * @param[out]  dest  result matrix
 */
CGLM_INLINE
void
glm_quat_mat4(versor q, mat4 dest) {
  float w, x, y, z,
        xx, yy, zz,
        xy, yz, xz,
        wx, wy, wz, norm, s;

  norm = glm_quat_norm(q);
  s    = norm > 0.0f ? 2.0f / norm : 0.0f;

  x = q[0];
  y = q[1];
  z = q[2];
  w = q[3];

  xx = s * x * x;   xy = s * x * y;   wx = s * w * x;
  yy = s * y * y;   yz = s * y * z;   wy = s * w * y;
  zz = s * z * z;   xz = s * x * z;   wz = s * w * z;

  dest[0][0] = 1.0f - yy - zz;
  dest[1][1] = 1.0f - xx - zz;
  dest[2][2] = 1.0f - xx - yy;

  dest[0][1] = xy + wz;
  dest[1][2] = yz + wx;
  dest[2][0] = xz + wy;

  dest[1][0] = xy - wz;
  dest[2][1] = yz - wx;
  dest[0][2] = xz - wy;

  dest[0][3] = 0.0f;
  dest[1][3] = 0.0f;
  dest[2][3] = 0.0f;
  dest[3][0] = 0.0f;
  dest[3][1] = 0.0f;
  dest[3][2] = 0.0f;
  dest[3][3] = 1.0f;
}

/*!
 * @brief convert quaternion to mat4 (transposed)
 *
 * @param[in]   q     quaternion
 * @param[out]  dest  result matrix as transposed
 */
CGLM_INLINE
void
glm_quat_mat4t(versor q, mat4 dest) {
  float w, x, y, z,
        xx, yy, zz,
        xy, yz, xz,
        wx, wy, wz, norm, s;

  norm = glm_quat_norm(q);
  s    = norm > 0.0f ? 2.0f / norm : 0.0f;

  x = q[0];
  y = q[1];
  z = q[2];
  w = q[3];

  xx = s * x * x;   xy = s * x * y;   wx = s * w * x;
  yy = s * y * y;   yz = s * y * z;   wy = s * w * y;
  zz = s * z * z;   xz = s * x * z;   wz = s * w * z;

  dest[0][0] = 1.0f - yy - zz;
  dest[1][1] = 1.0f - xx - zz;
  dest[2][2] = 1.0f - xx - yy;

  dest[1][0] = xy + wz;
  dest[2][1] = yz + wx;
  dest[0][2] = xz + wy;

  dest[0][1] = xy - wz;
  dest[1][2] = yz - wx;
  dest[2][0] = xz - wy;

  dest[0][3] = 0.0f;
  dest[1][3] = 0.0f;
  dest[2][3] = 0.0f;
  dest[3][0] = 0.0f;
  dest[3][1] = 0.0f;
  dest[3][2] = 0.0f;
  dest[3][3] = 1.0f;
}

/*!
 * @brief convert quaternion to mat3
 *
 * @param[in]   q     quaternion
 * @param[out]  dest  result matrix
 */
CGLM_INLINE
void
glm_quat_mat3(versor q, mat3 dest) {
  float w, x, y, z,
        xx, yy, zz,
        xy, yz, xz,
        wx, wy, wz, norm, s;

  norm = glm_quat_norm(q);
  s    = norm > 0.0f ? 2.0f / norm : 0.0f;

  x = q[0];
  y = q[1];
  z = q[2];
  w = q[3];

  xx = s * x * x;   xy = s * x * y;   wx = s * w * x;
  yy = s * y * y;   yz = s * y * z;   wy = s * w * y;
  zz = s * z * z;   xz = s * x * z;   wz = s * w * z;

  dest[0][0] = 1.0f - yy - zz;
  dest[1][1] = 1.0f - xx - zz;
  dest[2][2] = 1.0f - xx - yy;

  dest[0][1] = xy + wz;
  dest[1][2] = yz + wx;
  dest[2][0] = xz + wy;

  dest[1][0] = xy - wz;
  dest[2][1] = yz - wx;
  dest[0][2] = xz - wy;
}

/*!
 * @brief convert quaternion to mat3 (transposed)
 *
 * @param[in]   q     quaternion
 * @param[out]  dest  result matrix
 */
CGLM_INLINE
void
glm_quat_mat3t(versor q, mat3 dest) {
  float w, x, y, z,
        xx, yy, zz,
        xy, yz, xz,
        wx, wy, wz, norm, s;

  norm = glm_quat_norm(q);
  s    = norm > 0.0f ? 2.0f / norm : 0.0f;

  x = q[0];
  y = q[1];
  z = q[2];
  w = q[3];

  xx = s * x * x;   xy = s * x * y;   wx = s * w * x;
  yy = s * y * y;   yz = s * y * z;   wy = s * w * y;
  zz = s * z * z;   xz = s * x * z;   wz = s * w * z;

  dest[0][0] = 1.0f - yy - zz;
  dest[1][1] = 1.0f - xx - zz;
  dest[2][2] = 1.0f - xx - yy;

  dest[1][0] = xy + wz;
  dest[2][1] = yz + wx;
  dest[0][2] = xz + wy;

  dest[0][1] = xy - wz;
  dest[1][2] = yz - wx;
  dest[2][0] = xz - wy;
}

/*!
 * @brief interpolates between two quaternions
 *        using linear interpolation (LERP)
 *
 * @param[in]   from  from
 * @param[in]   to    to
 * @param[in]   t     interpolant (amount)
 * @param[out]  dest  result quaternion
 */
CGLM_INLINE
void
glm_quat_lerp(versor from, versor to, float t, versor dest) {
  glm_vec4_lerp(from, to, t, dest);
}

/*!
 * @brief interpolates between two quaternions
 *        using linear interpolation (LERP)
 *
 * @param[in]   from  from
 * @param[in]   to    to
 * @param[in]   t     interpolant (amount) clamped between 0 and 1
 * @param[out]  dest  result quaternion
 */
CGLM_INLINE
void
glm_quat_lerpc(versor from, versor to, float t, versor dest) {
  glm_vec4_lerpc(from, to, t, dest);
}

/*!
 * @brief interpolates between two quaternions
 *        using spherical linear interpolation (SLERP)
 *
 * @param[in]   from  from
 * @param[in]   to    to
 * @param[in]   t     amout
 * @param[out]  dest  result quaternion
 */
CGLM_INLINE
void
glm_quat_slerp(versor from, versor to, float t, versor dest) {
  CGLM_ALIGN(16) vec4 q1, q2;
  float cosTheta, sinTheta, angle;

  cosTheta = glm_quat_dot(from, to);
  glm_quat_copy(from, q1);

  if (fabsf(cosTheta) >= 1.0f) {
    glm_quat_copy(q1, dest);
    return;
  }

  if (cosTheta < 0.0f) {
    glm_vec4_negate(q1);
    cosTheta = -cosTheta;
  }

  sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

  /* LERP to avoid zero division */
  if (fabsf(sinTheta) < 0.001f) {
    glm_quat_lerp(from, to, t, dest);
    return;
  }

  /* SLERP */
  angle = acosf(cosTheta);
  glm_vec4_scale(q1, sinf((1.0f - t) * angle), q1);
  glm_vec4_scale(to, sinf(t * angle), q2);

  glm_vec4_add(q1, q2, q1);
  glm_vec4_scale(q1, 1.0f / sinTheta, dest);
}

/*!
 * @brief creates view matrix using quaternion as camera orientation
 *
 * @param[in]   eye   eye
 * @param[in]   ori   orientation in world space as quaternion
 * @param[out]  dest  view matrix
 */
CGLM_INLINE
void
glm_quat_look(vec3 eye, versor ori, mat4 dest) {
  /* orientation */
  glm_quat_mat4t(ori, dest);

  /* translate */
  glm_mat4_mulv3(dest, eye, 1.0f, dest[3]);
  glm_vec3_negate(dest[3]);
}

/*!
 * @brief creates look rotation quaternion
 *
 * @param[in]   dir   direction to look
 * @param[in]   up    up vector
 * @param[out]  dest  destination quaternion
 */
CGLM_INLINE
void
glm_quat_for(vec3 dir, vec3 up, versor dest) {
  CGLM_ALIGN_MAT mat3 m;

  glm_vec3_normalize_to(dir, m[2]); 

  /* No need to negate in LH, but we use RH here */
  glm_vec3_negate(m[2]);
  
  glm_vec3_crossn(up, m[2], m[0]);
  glm_vec3_cross(m[2], m[0], m[1]);

  glm_mat3_quat(m, dest);
}

/*!
 * @brief creates look rotation quaternion using source and
 *        destination positions p suffix stands for position
 *
 * @param[in]   from  source point
 * @param[in]   to    destination point
 * @param[in]   up    up vector
 * @param[out]  dest  destination quaternion
 */
CGLM_INLINE
void
glm_quat_forp(vec3 from, vec3 to, vec3 up, versor dest) {
  CGLM_ALIGN(8) vec3 dir;
  glm_vec3_sub(to, from, dir);
  glm_quat_for(dir, up, dest);
}

/*!
 * @brief rotate vector using using quaternion
 *
 * @param[in]   q     quaternion
 * @param[in]   v     vector to rotate
 * @param[out]  dest  rotated vector
 */
CGLM_INLINE
void
glm_quat_rotatev(versor q, vec3 v, vec3 dest) {
  CGLM_ALIGN(16) versor p;
  CGLM_ALIGN(8)  vec3   u, v1, v2;
  float s;

  glm_quat_normalize_to(q, p);
  glm_quat_imag(p, u);
  s = glm_quat_real(p);

  glm_vec3_scale(u, 2.0f * glm_vec3_dot(u, v), v1);
  glm_vec3_scale(v, s * s - glm_vec3_dot(u, u), v2);
  glm_vec3_add(v1, v2, v1);

  glm_vec3_cross(u, v, v2);
  glm_vec3_scale(v2, 2.0f * s, v2);

  glm_vec3_add(v1, v2, dest);
}

/*!
 * @brief rotate existing transform matrix using quaternion
 *
 * @param[in]   m     existing transform matrix
 * @param[in]   q     quaternion
 * @param[out]  dest  rotated matrix/transform
 */
CGLM_INLINE
void
glm_quat_rotate(mat4 m, versor q, mat4 dest) {
  CGLM_ALIGN_MAT mat4 rot;
  glm_quat_mat4(q, rot);
  glm_mul_rot(m, rot, dest);
}

/*!
 * @brief rotate existing transform matrix using quaternion at pivot point
 *
 * @param[in, out]   m     existing transform matrix
 * @param[in]        q     quaternion
 * @param[out]       pivot pivot
 */
CGLM_INLINE
void
glm_quat_rotate_at(mat4 m, versor q, vec3 pivot) {
  CGLM_ALIGN(8) vec3 pivotInv;

  glm_vec3_negate_to(pivot, pivotInv);

  glm_translate(m, pivot);
  glm_quat_rotate(m, q, m);
  glm_translate(m, pivotInv);
}

/*!
 * @brief rotate NEW transform matrix using quaternion at pivot point
 *
 * this creates rotation matrix, it assumes you don't have a matrix
 *
 * this should work faster than glm_quat_rotate_at because it reduces
 * one glm_translate.
 *
 * @param[out]  m     existing transform matrix
 * @param[in]   q     quaternion
 * @param[in]   pivot pivot
 */
CGLM_INLINE
void
glm_quat_rotate_atm(mat4 m, versor q, vec3 pivot) {
  CGLM_ALIGN(8) vec3 pivotInv;

  glm_vec3_negate_to(pivot, pivotInv);

  glm_translate_make(m, pivot);
  glm_quat_rotate(m, q, m);
  glm_translate(m, pivotInv);
}

#endif /* cglm_quat_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*
 NOTE:
  angles must be passed as [X-Angle, Y-Angle, Z-angle] order
  For instance you don't pass angles as [Z-Angle, X-Angle, Y-angle] to
  glm_euler_zxy funciton, All RELATED functions accept angles same order
  which is [X, Y, Z].
 */

/*
 Types:
   enum glm_euler_seq

 Functions:
   CGLM_INLINE glm_euler_seq glm_euler_order(int newOrder[3]);
   CGLM_INLINE void glm_euler_angles(mat4 m, vec3 dest);
   CGLM_INLINE void glm_euler(vec3 angles, mat4 dest);
   CGLM_INLINE void glm_euler_xyz(vec3 angles, mat4 dest);
   CGLM_INLINE void glm_euler_zyx(vec3 angles, mat4 dest);
   CGLM_INLINE void glm_euler_zxy(vec3 angles, mat4 dest);
   CGLM_INLINE void glm_euler_xzy(vec3 angles, mat4 dest);
   CGLM_INLINE void glm_euler_yzx(vec3 angles, mat4 dest);
   CGLM_INLINE void glm_euler_yxz(vec3 angles, mat4 dest);
   CGLM_INLINE void glm_euler_by_order(vec3         angles,
                                       glm_euler_seq ord,
                                       mat4         dest);
 */

#ifndef cglm_euler_h
#define cglm_euler_h


/*!
 * if you have axis order like vec3 orderVec = [0, 1, 2] or [0, 2, 1]...
 * vector then you can convert it to this enum by doing this:
 * @code
 * glm_euler_seq order;
 * order = orderVec[0] | orderVec[1] << 2 | orderVec[2] << 4;
 * @endcode
 * you may need to explicit cast if required
 */
typedef enum glm_euler_seq {
  GLM_EULER_XYZ = 0 << 0 | 1 << 2 | 2 << 4,
  GLM_EULER_XZY = 0 << 0 | 2 << 2 | 1 << 4,
  GLM_EULER_YZX = 1 << 0 | 2 << 2 | 0 << 4,
  GLM_EULER_YXZ = 1 << 0 | 0 << 2 | 2 << 4,
  GLM_EULER_ZXY = 2 << 0 | 0 << 2 | 1 << 4,
  GLM_EULER_ZYX = 2 << 0 | 1 << 2 | 0 << 4
} glm_euler_seq;

CGLM_INLINE
glm_euler_seq
glm_euler_order(int ord[3]) {
  return (glm_euler_seq)(ord[0] << 0 | ord[1] << 2 | ord[2] << 4);
}

/*!
 * @brief extract euler angles (in radians) using xyz order
 *
 * @param[in]  m    affine transform
 * @param[out] dest angles vector [x, y, z]
 */
CGLM_INLINE
void
glm_euler_angles(mat4 m, vec3 dest) {
  float m00, m01, m10, m11, m20, m21, m22;
  float thetaX, thetaY, thetaZ;

  m00 = m[0][0];  m10 = m[1][0];  m20 = m[2][0];
  m01 = m[0][1];  m11 = m[1][1];  m21 = m[2][1];
                                  m22 = m[2][2];

  if (m20 < 1.0f) {
    if (m20 > -1.0f) {
      thetaY = asinf(m20);
      thetaX = atan2f(-m21, m22);
      thetaZ = atan2f(-m10, m00);
    } else { /* m20 == -1 */
      /* Not a unique solution */
      thetaY = -GLM_PI_2f;
      thetaX = -atan2f(m01, m11);
      thetaZ =  0.0f;
    }
  } else { /* m20 == +1 */
    thetaY = GLM_PI_2f;
    thetaX = atan2f(m01, m11);
    thetaZ = 0.0f;
  }

  dest[0] = thetaX;
  dest[1] = thetaY;
  dest[2] = thetaZ;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[out] dest   rotation matrix
 */
CGLM_INLINE
void
glm_euler_xyz(vec3 angles, mat4 dest) {
  float cx, cy, cz,
        sx, sy, sz, czsx, cxcz, sysz;

  sx   = sinf(angles[0]); cx = cosf(angles[0]);
  sy   = sinf(angles[1]); cy = cosf(angles[1]);
  sz   = sinf(angles[2]); cz = cosf(angles[2]);

  czsx = cz * sx;
  cxcz = cx * cz;
  sysz = sy * sz;

  dest[0][0] =  cy * cz;
  dest[0][1] =  czsx * sy + cx * sz;
  dest[0][2] = -cxcz * sy + sx * sz;
  dest[1][0] = -cy * sz;
  dest[1][1] =  cxcz - sx * sysz;
  dest[1][2] =  czsx + cx * sysz;
  dest[2][0] =  sy;
  dest[2][1] = -cy * sx;
  dest[2][2] =  cx * cy;
  dest[0][3] =  0.0f;
  dest[1][3] =  0.0f;
  dest[2][3] =  0.0f;
  dest[3][0] =  0.0f;
  dest[3][1] =  0.0f;
  dest[3][2] =  0.0f;
  dest[3][3] =  1.0f;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[out] dest   rotation matrix
 */
CGLM_INLINE
void
glm_euler(vec3 angles, mat4 dest) {
  glm_euler_xyz(angles, dest);
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[out] dest   rotation matrix
 */
CGLM_INLINE
void
glm_euler_xzy(vec3 angles, mat4 dest) {
  float cx, cy, cz,
  sx, sy, sz, sxsy, cysx, cxsy, cxcy;

  sx   = sinf(angles[0]); cx = cosf(angles[0]);
  sy   = sinf(angles[1]); cy = cosf(angles[1]);
  sz   = sinf(angles[2]); cz = cosf(angles[2]);

  sxsy = sx * sy;
  cysx = cy * sx;
  cxsy = cx * sy;
  cxcy = cx * cy;

  dest[0][0] =  cy * cz;
  dest[0][1] =  sxsy + cxcy * sz;
  dest[0][2] = -cxsy + cysx * sz;
  dest[1][0] = -sz;
  dest[1][1] =  cx * cz;
  dest[1][2] =  cz * sx;
  dest[2][0] =  cz * sy;
  dest[2][1] = -cysx + cxsy * sz;
  dest[2][2] =  cxcy + sxsy * sz;
  dest[0][3] =  0.0f;
  dest[1][3] =  0.0f;
  dest[2][3] =  0.0f;
  dest[3][0] =  0.0f;
  dest[3][1] =  0.0f;
  dest[3][2] =  0.0f;
  dest[3][3] =  1.0f;
}


/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[out] dest   rotation matrix
 */
CGLM_INLINE
void
glm_euler_yxz(vec3 angles, mat4 dest) {
  float cx, cy, cz,
        sx, sy, sz, cycz, sysz, czsy, cysz;

  sx   = sinf(angles[0]); cx = cosf(angles[0]);
  sy   = sinf(angles[1]); cy = cosf(angles[1]);
  sz   = sinf(angles[2]); cz = cosf(angles[2]);

  cycz = cy * cz;
  sysz = sy * sz;
  czsy = cz * sy;
  cysz = cy * sz;

  dest[0][0] =  cycz + sx * sysz;
  dest[0][1] =  cx * sz;
  dest[0][2] = -czsy + cysz * sx;
  dest[1][0] = -cysz + czsy * sx;
  dest[1][1] =  cx * cz;
  dest[1][2] =  cycz * sx + sysz;
  dest[2][0] =  cx * sy;
  dest[2][1] = -sx;
  dest[2][2] =  cx * cy;
  dest[0][3] =  0.0f;
  dest[1][3] =  0.0f;
  dest[2][3] =  0.0f;
  dest[3][0] =  0.0f;
  dest[3][1] =  0.0f;
  dest[3][2] =  0.0f;
  dest[3][3] =  1.0f;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[out] dest   rotation matrix
 */
CGLM_INLINE
void
glm_euler_yzx(vec3 angles, mat4 dest) {
  float cx, cy, cz,
        sx, sy, sz, sxsy, cxcy, cysx, cxsy;

  sx   = sinf(angles[0]); cx = cosf(angles[0]);
  sy   = sinf(angles[1]); cy = cosf(angles[1]);
  sz   = sinf(angles[2]); cz = cosf(angles[2]);

  sxsy = sx * sy;
  cxcy = cx * cy;
  cysx = cy * sx;
  cxsy = cx * sy;

  dest[0][0] =  cy * cz;
  dest[0][1] =  sz;
  dest[0][2] = -cz * sy;
  dest[1][0] =  sxsy - cxcy * sz;
  dest[1][1] =  cx * cz;
  dest[1][2] =  cysx + cxsy * sz;
  dest[2][0] =  cxsy + cysx * sz;
  dest[2][1] = -cz * sx;
  dest[2][2] =  cxcy - sxsy * sz;
  dest[0][3] =  0.0f;
  dest[1][3] =  0.0f;
  dest[2][3] =  0.0f;
  dest[3][0] =  0.0f;
  dest[3][1] =  0.0f;
  dest[3][2] =  0.0f;
  dest[3][3] =  1.0f;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[out] dest   rotation matrix
 */
CGLM_INLINE
void
glm_euler_zxy(vec3 angles, mat4 dest) {
  float cx, cy, cz,
        sx, sy, sz, cycz, sxsy, cysz;

  sx   = sinf(angles[0]); cx = cosf(angles[0]);
  sy   = sinf(angles[1]); cy = cosf(angles[1]);
  sz   = sinf(angles[2]); cz = cosf(angles[2]);

  cycz = cy * cz;
  sxsy = sx * sy;
  cysz = cy * sz;

  dest[0][0] =  cycz - sxsy * sz;
  dest[0][1] =  cz * sxsy + cysz;
  dest[0][2] = -cx * sy;
  dest[1][0] = -cx * sz;
  dest[1][1] =  cx * cz;
  dest[1][2] =  sx;
  dest[2][0] =  cz * sy + cysz * sx;
  dest[2][1] = -cycz * sx + sy * sz;
  dest[2][2] =  cx * cy;
  dest[0][3] =  0.0f;
  dest[1][3] =  0.0f;
  dest[2][3] =  0.0f;
  dest[3][0] =  0.0f;
  dest[3][1] =  0.0f;
  dest[3][2] =  0.0f;
  dest[3][3] =  1.0f;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[out] dest   rotation matrix
 */
CGLM_INLINE
void
glm_euler_zyx(vec3 angles, mat4 dest) {
  float cx, cy, cz,
        sx, sy, sz, czsx, cxcz, sysz;

  sx   = sinf(angles[0]); cx = cosf(angles[0]);
  sy   = sinf(angles[1]); cy = cosf(angles[1]);
  sz   = sinf(angles[2]); cz = cosf(angles[2]);

  czsx = cz * sx;
  cxcz = cx * cz;
  sysz = sy * sz;

  dest[0][0] =  cy * cz;
  dest[0][1] =  cy * sz;
  dest[0][2] = -sy;
  dest[1][0] =  czsx * sy - cx * sz;
  dest[1][1] =  cxcz + sx * sysz;
  dest[1][2] =  cy * sx;
  dest[2][0] =  cxcz * sy + sx * sz;
  dest[2][1] = -czsx + cx * sysz;
  dest[2][2] =  cx * cy;
  dest[0][3] =  0.0f;
  dest[1][3] =  0.0f;
  dest[2][3] =  0.0f;
  dest[3][0] =  0.0f;
  dest[3][1] =  0.0f;
  dest[3][2] =  0.0f;
  dest[3][3] =  1.0f;
}

/*!
 * @brief build rotation matrix from euler angles
 *
 * @param[in]  angles angles as vector [Xangle, Yangle, Zangle]
 * @param[in]  ord    euler order
 * @param[out] dest   rotation matrix
 */
CGLM_INLINE
void
glm_euler_by_order(vec3 angles, glm_euler_seq ord, mat4 dest) {
  float cx, cy, cz,
        sx, sy, sz;

  float cycz, cysz, cysx, cxcy,
        czsy, cxcz, czsx, cxsz,
        sysz;

  sx = sinf(angles[0]); cx = cosf(angles[0]);
  sy = sinf(angles[1]); cy = cosf(angles[1]);
  sz = sinf(angles[2]); cz = cosf(angles[2]);

  cycz = cy * cz; cysz = cy * sz;
  cysx = cy * sx; cxcy = cx * cy;
  czsy = cz * sy; cxcz = cx * cz;
  czsx = cz * sx; cxsz = cx * sz;
  sysz = sy * sz;

  switch (ord) {
    case GLM_EULER_XZY:
      dest[0][0] =  cycz;
      dest[0][1] =  sx * sy + cx * cysz;
      dest[0][2] = -cx * sy + cysx * sz;
      dest[1][0] = -sz;
      dest[1][1] =  cxcz;
      dest[1][2] =  czsx;
      dest[2][0] =  czsy;
      dest[2][1] = -cysx + cx * sysz;
      dest[2][2] =  cxcy + sx * sysz;
      break;
    case GLM_EULER_XYZ:
      dest[0][0] =  cycz;
      dest[0][1] =  czsx * sy + cxsz;
      dest[0][2] = -cx * czsy + sx * sz;
      dest[1][0] = -cysz;
      dest[1][1] =  cxcz - sx * sysz;
      dest[1][2] =  czsx + cx * sysz;
      dest[2][0] =  sy;
      dest[2][1] = -cysx;
      dest[2][2] =  cxcy;
      break;
    case GLM_EULER_YXZ:
      dest[0][0] =  cycz + sx * sysz;
      dest[0][1] =  cxsz;
      dest[0][2] = -czsy + cysx * sz;
      dest[1][0] =  czsx * sy - cysz;
      dest[1][1] =  cxcz;
      dest[1][2] =  cycz * sx + sysz;
      dest[2][0] =  cx * sy;
      dest[2][1] = -sx;
      dest[2][2] =  cxcy;
      break;
    case GLM_EULER_YZX:
      dest[0][0] =  cycz;
      dest[0][1] =  sz;
      dest[0][2] = -czsy;
      dest[1][0] =  sx * sy - cx * cysz;
      dest[1][1] =  cxcz;
      dest[1][2] =  cysx + cx * sysz;
      dest[2][0] =  cx * sy + cysx * sz;
      dest[2][1] = -czsx;
      dest[2][2] =  cxcy - sx * sysz;
      break;
    case GLM_EULER_ZXY:
      dest[0][0] =  cycz - sx * sysz;
      dest[0][1] =  czsx * sy + cysz;
      dest[0][2] = -cx * sy;
      dest[1][0] = -cxsz;
      dest[1][1] =  cxcz;
      dest[1][2] =  sx;
      dest[2][0] =  czsy + cysx * sz;
      dest[2][1] = -cycz * sx + sysz;
      dest[2][2] =  cxcy;
      break;
    case GLM_EULER_ZYX:
      dest[0][0] =  cycz;
      dest[0][1] =  cysz;
      dest[0][2] = -sy;
      dest[1][0] =  czsx * sy - cxsz;
      dest[1][1] =  cxcz + sx * sysz;
      dest[1][2] =  cysx;
      dest[2][0] =  cx * czsy + sx * sz;
      dest[2][1] = -czsx + cx * sysz;
      dest[2][2] =  cxcy;
      break;
  }

  dest[0][3] = 0.0f;
  dest[1][3] = 0.0f;
  dest[2][3] = 0.0f;
  dest[3][0] = 0.0f;
  dest[3][1] = 0.0f;
  dest[3][2] = 0.0f;
  dest[3][3] = 1.0f;
}

#endif /* cglm_euler_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_box_h
#define cglm_box_h


/*!
 * @brief apply transform to Axis-Aligned Bounding Box
 *
 * @param[in]  box  bounding box
 * @param[in]  m    transform matrix
 * @param[out] dest transformed bounding box
 */
CGLM_INLINE
void
glm_aabb_transform(vec3 box[2], mat4 m, vec3 dest[2]) {
  vec3 v[2], xa, xb, ya, yb, za, zb;

  glm_vec3_scale(m[0], box[0][0], xa);
  glm_vec3_scale(m[0], box[1][0], xb);

  glm_vec3_scale(m[1], box[0][1], ya);
  glm_vec3_scale(m[1], box[1][1], yb);

  glm_vec3_scale(m[2], box[0][2], za);
  glm_vec3_scale(m[2], box[1][2], zb);

  /* translation + min(xa, xb) + min(ya, yb) + min(za, zb) */
  glm_vec3(m[3], v[0]);
  glm_vec3_minadd(xa, xb, v[0]);
  glm_vec3_minadd(ya, yb, v[0]);
  glm_vec3_minadd(za, zb, v[0]);

  /* translation + max(xa, xb) + max(ya, yb) + max(za, zb) */
  glm_vec3(m[3], v[1]);
  glm_vec3_maxadd(xa, xb, v[1]);
  glm_vec3_maxadd(ya, yb, v[1]);
  glm_vec3_maxadd(za, zb, v[1]);

  glm_vec3_copy(v[0], dest[0]);
  glm_vec3_copy(v[1], dest[1]);
}

/*!
 * @brief merges two AABB bounding box and creates new one
 *
 * two box must be in same space, if one of box is in different space then
 * you should consider to convert it's space by glm_box_space
 *
 * @param[in]  box1 bounding box 1
 * @param[in]  box2 bounding box 2
 * @param[out] dest merged bounding box
 */
CGLM_INLINE
void
glm_aabb_merge(vec3 box1[2], vec3 box2[2], vec3 dest[2]) {
  dest[0][0] = glm_min(box1[0][0], box2[0][0]);
  dest[0][1] = glm_min(box1[0][1], box2[0][1]);
  dest[0][2] = glm_min(box1[0][2], box2[0][2]);

  dest[1][0] = glm_max(box1[1][0], box2[1][0]);
  dest[1][1] = glm_max(box1[1][1], box2[1][1]);
  dest[1][2] = glm_max(box1[1][2], box2[1][2]);
}

/*!
 * @brief crops a bounding box with another one.
 *
 * this could be useful for gettng a bbox which fits with view frustum and
 * object bounding boxes. In this case you crop view frustum box with objects
 * box
 *
 * @param[in]  box     bounding box 1
 * @param[in]  cropBox crop box
 * @param[out] dest    cropped bounding box
 */
CGLM_INLINE
void
glm_aabb_crop(vec3 box[2], vec3 cropBox[2], vec3 dest[2]) {
  dest[0][0] = glm_max(box[0][0], cropBox[0][0]);
  dest[0][1] = glm_max(box[0][1], cropBox[0][1]);
  dest[0][2] = glm_max(box[0][2], cropBox[0][2]);

  dest[1][0] = glm_min(box[1][0], cropBox[1][0]);
  dest[1][1] = glm_min(box[1][1], cropBox[1][1]);
  dest[1][2] = glm_min(box[1][2], cropBox[1][2]);
}

/*!
 * @brief crops a bounding box with another one.
 *
 * this could be useful for gettng a bbox which fits with view frustum and
 * object bounding boxes. In this case you crop view frustum box with objects
 * box
 *
 * @param[in]  box      bounding box
 * @param[in]  cropBox  crop box
 * @param[in]  clampBox miniumum box
 * @param[out] dest     cropped bounding box
 */
CGLM_INLINE
void
glm_aabb_crop_until(vec3 box[2],
                    vec3 cropBox[2],
                    vec3 clampBox[2],
                    vec3 dest[2]) {
  glm_aabb_crop(box, cropBox, dest);
  glm_aabb_merge(clampBox, dest, dest);
}

/*!
 * @brief check if AABB intersects with frustum planes
 *
 * this could be useful for frustum culling using AABB.
 *
 * OPTIMIZATION HINT:
 *  if planes order is similar to LEFT, RIGHT, BOTTOM, TOP, NEAR, FAR
 *  then this method should run even faster because it would only use two
 *  planes if object is not inside the two planes
 *  fortunately cglm extracts planes as this order! just pass what you got!
 *
 * @param[in]  box     bounding box
 * @param[in]  planes  frustum planes
 */
CGLM_INLINE
bool
glm_aabb_frustum(vec3 box[2], vec4 planes[6]) {
  float *p, dp;
  int    i;

  for (i = 0; i < 6; i++) {
    p  = planes[i];
    dp = p[0] * box[p[0] > 0.0f][0]
       + p[1] * box[p[1] > 0.0f][1]
       + p[2] * box[p[2] > 0.0f][2];

    if (dp < -p[3])
      return false;
  }

  return true;
}

/*!
 * @brief invalidate AABB min and max values
 *
 * @param[in, out]  box bounding box
 */
CGLM_INLINE
void
glm_aabb_invalidate(vec3 box[2]) {
  glm_vec3_broadcast(FLT_MAX,  box[0]);
  glm_vec3_broadcast(-FLT_MAX, box[1]);
}

/*!
 * @brief check if AABB is valid or not
 *
 * @param[in]  box bounding box
 */
CGLM_INLINE
bool
glm_aabb_isvalid(vec3 box[2]) {
  return glm_vec3_max(box[0]) != FLT_MAX
         && glm_vec3_min(box[1]) != -FLT_MAX;
}

/*!
 * @brief distance between of min and max
 *
 * @param[in]  box bounding box
 */
CGLM_INLINE
float
glm_aabb_size(vec3 box[2]) {
  return glm_vec3_distance(box[0], box[1]);
}

/*!
 * @brief radius of sphere which surrounds AABB
 *
 * @param[in]  box bounding box
 */
CGLM_INLINE
float
glm_aabb_radius(vec3 box[2]) {
  return glm_aabb_size(box) * 0.5f;
}

/*!
 * @brief computes center point of AABB
 *
 * @param[in]   box  bounding box
 * @param[out]  dest center of bounding box
 */
CGLM_INLINE
void
glm_aabb_center(vec3 box[2], vec3 dest) {
  glm_vec3_center(box[0], box[1], dest);
}

/*!
 * @brief check if two AABB intersects
 *
 * @param[in]   box    bounding box
 * @param[in]   other  other bounding box
 */
CGLM_INLINE
bool
glm_aabb_aabb(vec3 box[2], vec3 other[2]) {
  return (box[0][0] <= other[1][0] && box[1][0] >= other[0][0])
      && (box[0][1] <= other[1][1] && box[1][1] >= other[0][1])
      && (box[0][2] <= other[1][2] && box[1][2] >= other[0][2]);
}

/*!
 * @brief check if AABB intersects with sphere
 *
 * https://github.com/erich666/GraphicsGems/blob/master/gems/BoxSphere.c
 * Solid Box - Solid Sphere test.
 *
 * @param[in]   box    solid bounding box
 * @param[in]   s      solid sphere
 */
CGLM_INLINE
bool
glm_aabb_sphere(vec3 box[2], vec4 s) {
  float dmin;
  int   a, b, c;

  a = s[0] >= box[0][0];
  b = s[1] >= box[0][1];
  c = s[2] >= box[0][2];

  dmin  = glm_pow2(s[0] - box[a][0])
        + glm_pow2(s[1] - box[b][1])
        + glm_pow2(s[2] - box[c][2]);

  return dmin <= glm_pow2(s[3]);
}

/*!
 * @brief check if point is inside of AABB
 *
 * @param[in]   box    bounding box
 * @param[in]   point  point
 */
CGLM_INLINE
bool
glm_aabb_point(vec3 box[2], vec3 point) {
  return (point[0] >= box[0][0] && point[0] <= box[1][0])
      && (point[1] >= box[0][1] && point[1] <= box[1][1])
      && (point[2] >= box[0][2] && point[2] <= box[1][2]);
}

/*!
 * @brief check if AABB contains other AABB
 *
 * @param[in]   box    bounding box
 * @param[in]   other  other bounding box
 */
CGLM_INLINE
bool
glm_aabb_contains(vec3 box[2], vec3 other[2]) {
  return (box[0][0] <= other[0][0] && box[1][0] >= other[1][0])
      && (box[0][1] <= other[0][1] && box[1][1] >= other[1][1])
      && (box[0][2] <= other[0][2] && box[1][2] >= other[1][2]);
}

#endif /* cglm_box_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_color_h
#define cglm_color_h


/*!
 * @brief averages the color channels into one value
 *
 * @param[in]  rgb RGB color
 */
CGLM_INLINE
float
glm_luminance(vec3 rgb) {
  vec3 l = {0.212671f, 0.715160f, 0.072169f};
  return glm_dot(rgb, l);
}

#endif /* cglm_color_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*
 Functions:
   CGLM_INLINE void glm_mat4_print(mat4 matrix, FILE *ostream);
   CGLM_INLINE void glm_mat3_print(mat3 matrix, FILE *ostream);
   CGLM_INLINE void glm_vec4_print(vec4 vec, FILE *ostream);
   CGLM_INLINE void glm_vec3_print(vec3 vec, FILE *ostream);
   CGLM_INLINE void glm_ivec3_print(ivec3 vec, FILE *ostream);
   CGLM_INLINE void glm_versor_print(versor vec, FILE *ostream);
 */

/*
 cglm tried to enable print functions in debug mode and disable them in
 release/production mode to eliminate printing costs.
 
 if you need to force enable then define CGLM_DEFINE_PRINTS macro not DEBUG one
 
 Print functions are enabled if:
 
 - DEBUG or _DEBUG macro is defined (mostly defined automatically in debugging)
 - CGLM_DEFINE_PRINTS macro is defined including release/production
   which makes enabled printing always
 - glmc_ calls for io are always prints

 */

/* DEPRECATED: CGLM_NO_PRINTS_NOOP (use CGLM_DEFINE_PRINTS) */

#ifndef cglm_io_h
#define cglm_io_h
#if defined(DEBUG) || defined(_DEBUG) \
   || defined(CGLM_DEFINE_PRINTS) || defined(CGLM_LIB_SRC) \
   || defined(CGLM_NO_PRINTS_NOOP)


#include <stdio.h>
#include <stdlib.h>

#ifndef CGLM_PRINT_PRECISION
#  define CGLM_PRINT_PRECISION    5
#endif

#ifndef CGLM_PRINT_MAX_TO_SHORT
#  define CGLM_PRINT_MAX_TO_SHORT 1e5
#endif

#ifndef CGLM_PRINT_COLOR
#  define CGLM_PRINT_COLOR        "\033[36m"
#endif

#ifndef CGLM_PRINT_COLOR_RESET
#  define CGLM_PRINT_COLOR_RESET  "\033[0m"
#endif

CGLM_INLINE
void
glm_mat4_print(mat4              matrix,
               FILE * __restrict ostream) {
  char buff[16];
  int  i, j, cw[4], cwi;

#define m 4
#define n 4

  fprintf(ostream, "Matrix (float%dx%d): " CGLM_PRINT_COLOR "\n" , m, n);

  cw[0] = cw[1] = cw[2] = cw[3] = 0;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      if (matrix[i][j] < CGLM_PRINT_MAX_TO_SHORT)
        cwi = sprintf(buff, "% .*f", CGLM_PRINT_PRECISION, matrix[i][j]);
      else
        cwi = sprintf(buff, "% g", matrix[i][j]);
      cw[i] = GLM_MAX(cw[i], cwi);
    }
  }

  for (i = 0; i < m; i++) {
    fprintf(ostream, "  |");

    for (j = 0; j < n; j++)
      if (matrix[i][j] < CGLM_PRINT_MAX_TO_SHORT)
        fprintf(ostream, " % *.*f", cw[j], CGLM_PRINT_PRECISION, matrix[j][i]);
      else
        fprintf(ostream, " % *g", cw[j], matrix[j][i]);

    fprintf(ostream, "  |\n");
  }

  fprintf(ostream, CGLM_PRINT_COLOR_RESET "\n");

#undef m
#undef n
}


CGLM_INLINE
void
glm_mat3_print(mat3              matrix,
               FILE * __restrict ostream) {
  char buff[16];
  int  i, j, cw[4], cwi;

#define m 3
#define n 3

  fprintf(ostream, "Matrix (float%dx%d): " CGLM_PRINT_COLOR "\n", m, n);

  cw[0] = cw[1] = cw[2] = 0;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      if (matrix[i][j] < CGLM_PRINT_MAX_TO_SHORT)
        cwi = sprintf(buff, "% .*f", CGLM_PRINT_PRECISION, matrix[i][j]);
      else
        cwi = sprintf(buff, "% g", matrix[i][j]);
      cw[i] = GLM_MAX(cw[i], cwi);
    }
  }

  for (i = 0; i < m; i++) {
    fprintf(ostream, "  |");
    
    for (j = 0; j < n; j++)
      if (matrix[i][j] < CGLM_PRINT_MAX_TO_SHORT)
        fprintf(ostream, " % *.*f", cw[j], CGLM_PRINT_PRECISION, matrix[j][i]);
      else
        fprintf(ostream, " % *g", cw[j], matrix[j][i]);
    
    fprintf(ostream, "  |\n");
  }

  fprintf(ostream, CGLM_PRINT_COLOR_RESET "\n");

#undef m
#undef n
}

CGLM_INLINE
void
glm_mat2_print(mat2              matrix,
               FILE * __restrict ostream) {
  char buff[16];
  int  i, j, cw[4], cwi;

#define m 2
#define n 2

  fprintf(ostream, "Matrix (float%dx%d): " CGLM_PRINT_COLOR "\n", m, n);

  cw[0] = cw[1] = 0;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      if (matrix[i][j] < CGLM_PRINT_MAX_TO_SHORT)
        cwi = sprintf(buff, "% .*f", CGLM_PRINT_PRECISION, matrix[i][j]);
      else
        cwi = sprintf(buff, "% g", matrix[i][j]);
      cw[i] = GLM_MAX(cw[i], cwi);
    }
  }

  for (i = 0; i < m; i++) {
    fprintf(ostream, "  |");
    
    for (j = 0; j < n; j++)
      if (matrix[i][j] < CGLM_PRINT_MAX_TO_SHORT)
        fprintf(ostream, " % *.*f", cw[j], CGLM_PRINT_PRECISION, matrix[j][i]);
      else
        fprintf(ostream, " % *g", cw[j], matrix[j][i]);
    
    fprintf(ostream, "  |\n");
  }

  fprintf(ostream, CGLM_PRINT_COLOR_RESET "\n");

#undef m
#undef n
}

CGLM_INLINE
void
glm_vec4_print(vec4              vec,
               FILE * __restrict ostream) {
  int i;

#define m 4

  fprintf(ostream, "Vector (float%d): " CGLM_PRINT_COLOR "\n  (", m);

  for (i = 0; i < m; i++) {
    if (vec[i] < CGLM_PRINT_MAX_TO_SHORT)
      fprintf(ostream, " % .*f", CGLM_PRINT_PRECISION, vec[i]);
    else
      fprintf(ostream, " % g", vec[i]);
  }

  fprintf(ostream, "  )" CGLM_PRINT_COLOR_RESET "\n\n");

#undef m
}

CGLM_INLINE
void
glm_vec3_print(vec3              vec,
               FILE * __restrict ostream) {
  int i;

#define m 3

  fprintf(ostream, "Vector (float%d): " CGLM_PRINT_COLOR "\n  (", m);

  for (i = 0; i < m; i++) {
    if (vec[i] < CGLM_PRINT_MAX_TO_SHORT)
      fprintf(ostream, " % .*f", CGLM_PRINT_PRECISION, vec[i]);
    else
      fprintf(ostream, " % g", vec[i]);
  }

  fprintf(ostream, "  )" CGLM_PRINT_COLOR_RESET "\n\n");

#undef m
}

CGLM_INLINE
void
glm_ivec3_print(ivec3             vec,
                FILE * __restrict ostream) {
  int i;

#define m 3

  fprintf(ostream, "Vector (int%d): " CGLM_PRINT_COLOR "\n  (", m);

  for (i = 0; i < m; i++)
    fprintf(ostream, " % d", vec[i]);

  fprintf(ostream, "  )" CGLM_PRINT_COLOR_RESET "\n\n");
  
#undef m
}

CGLM_INLINE
void
glm_vec2_print(vec2              vec,
               FILE * __restrict ostream) {
  int i;

#define m 2

  fprintf(ostream, "Vector (float%d): " CGLM_PRINT_COLOR "\n  (", m);

  for (i = 0; i < m; i++) {
    if (vec[i] < CGLM_PRINT_MAX_TO_SHORT)
      fprintf(ostream, " % .*f", CGLM_PRINT_PRECISION, vec[i]);
    else
      fprintf(ostream, " % g", vec[i]);
  }

  fprintf(ostream, "  )" CGLM_PRINT_COLOR_RESET "\n\n");

#undef m
}

CGLM_INLINE
void
glm_versor_print(versor            vec,
                 FILE * __restrict ostream) {
  int i;

#define m 4

  fprintf(ostream, "Quaternion (float%d): " CGLM_PRINT_COLOR "\n  (", m);

  for (i = 0; i < m; i++) {
    if (vec[i] < CGLM_PRINT_MAX_TO_SHORT)
      fprintf(ostream, " % .*f", CGLM_PRINT_PRECISION, vec[i]);
    else
      fprintf(ostream, " % g", vec[i]);
  }


  fprintf(ostream, "  )" CGLM_PRINT_COLOR_RESET "\n\n");

#undef m
}

CGLM_INLINE
void
glm_aabb_print(vec3                    bbox[2],
               const char * __restrict tag,
               FILE       * __restrict ostream) {
  int i, j;

#define m 3

  fprintf(ostream, "AABB (%s): " CGLM_PRINT_COLOR "\n", tag ? tag: "float");

  for (i = 0; i < 2; i++) {
    fprintf(ostream, "  (");
    
    for (j = 0; j < m; j++) {
      if (bbox[i][j] < CGLM_PRINT_MAX_TO_SHORT)
        fprintf(ostream, " % .*f", CGLM_PRINT_PRECISION, bbox[i][j]);
      else
        fprintf(ostream, " % g", bbox[i][j]);
    }

    fprintf(ostream, "  )\n");
  }

  fprintf(ostream, CGLM_PRINT_COLOR_RESET "\n");

#undef m
}

#else


#include <stdio.h>
#include <stdlib.h>

/* NOOP: Remove print from DEBUG */
#define glm_mat4_print(v, s) (void)v; (void)s;
#define glm_mat3_print(v, s) (void)v; (void)s;
#define glm_mat2_print(v, s) (void)v; (void)s;
#define glm_vec4_print(v, s) (void)v; (void)s;
#define glm_vec3_print(v, s) (void)v; (void)s;
#define glm_ivec3_print(v, s) (void)v; (void)s;
#define glm_vec2_print(v, s) (void)v; (void)s;
#define glm_versor_print(v, s) (void)v; (void)s;
#define glm_aabb_print(v, t, s) (void)v; (void)t; (void)s;

#endif
#endif /* cglm_io_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_project_h
#define cglm_project_h


/*!
 * @brief maps the specified viewport coordinates into specified space [1]
 *        the matrix should contain projection matrix.
 *
 * if you don't have ( and don't want to have ) an inverse matrix then use
 * glm_unproject version. You may use existing inverse of matrix in somewhere
 * else, this is why glm_unprojecti exists to save save inversion cost
 *
 * [1] space:
 *  1- if m = invProj:     View Space
 *  2- if m = invViewProj: World Space
 *  3- if m = invMVP:      Object Space
 *
 * You probably want to map the coordinates into object space
 * so use invMVP as m
 *
 * Computing viewProj:
 *   glm_mat4_mul(proj, view, viewProj);
 *   glm_mat4_mul(viewProj, model, MVP);
 *   glm_mat4_inv(viewProj, invMVP);
 *
 * @param[in]  pos      point/position in viewport coordinates
 * @param[in]  invMat   matrix (see brief)
 * @param[in]  vp       viewport as [x, y, width, height]
 * @param[out] dest     unprojected coordinates
 */
CGLM_INLINE
void
glm_unprojecti(vec3 pos, mat4 invMat, vec4 vp, vec3 dest) {
  vec4 v;

  v[0] = 2.0f * (pos[0] - vp[0]) / vp[2] - 1.0f;
  v[1] = 2.0f * (pos[1] - vp[1]) / vp[3] - 1.0f;
  v[2] = 2.0f *  pos[2]                  - 1.0f;
  v[3] = 1.0f;

  glm_mat4_mulv(invMat, v, v);
  glm_vec4_scale(v, 1.0f / v[3], v);
  glm_vec3(v, dest);
}

/*!
 * @brief maps the specified viewport coordinates into specified space [1]
 *        the matrix should contain projection matrix.
 *
 * this is same as glm_unprojecti except this function get inverse matrix for
 * you.
 *
 * [1] space:
 *  1- if m = proj:     View Space
 *  2- if m = viewProj: World Space
 *  3- if m = MVP:      Object Space
 *
 * You probably want to map the coordinates into object space
 * so use MVP as m
 *
 * Computing viewProj and MVP:
 *   glm_mat4_mul(proj, view, viewProj);
 *   glm_mat4_mul(viewProj, model, MVP);
 *
 * @param[in]  pos      point/position in viewport coordinates
 * @param[in]  m        matrix (see brief)
 * @param[in]  vp       viewport as [x, y, width, height]
 * @param[out] dest     unprojected coordinates
 */
CGLM_INLINE
void
glm_unproject(vec3 pos, mat4 m, vec4 vp, vec3 dest) {
  mat4 inv;
  glm_mat4_inv(m, inv);
  glm_unprojecti(pos, inv, vp, dest);
}

/*!
 * @brief map object coordinates to window coordinates
 *
 * Computing MVP:
 *   glm_mat4_mul(proj, view, viewProj);
 *   glm_mat4_mul(viewProj, model, MVP);
 *
 * @param[in]  pos      object coordinates
 * @param[in]  m        MVP matrix
 * @param[in]  vp       viewport as [x, y, width, height]
 * @param[out] dest     projected coordinates
 */
CGLM_INLINE
void
glm_project(vec3 pos, mat4 m, vec4 vp, vec3 dest) {
  CGLM_ALIGN(16) vec4 pos4, vone = GLM_VEC4_ONE_INIT;

  glm_vec4(pos, 1.0f, pos4);

  glm_mat4_mulv(m, pos4, pos4);
  glm_vec4_scale(pos4, 1.0f / pos4[3], pos4); /* pos = pos / pos.w */
  glm_vec4_add(pos4, vone, pos4);
  glm_vec4_scale(pos4, 0.5f, pos4);

  dest[0] = pos4[0] * vp[2] + vp[0];
  dest[1] = pos4[1] * vp[3] + vp[1];
  dest[2] = pos4[2];
}

#endif /* cglm_project_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_sphere_h
#define cglm_sphere_h


/*
  Sphere Representation in cglm: [center.x, center.y, center.z, radii]

  You could use this representation or you can convert it to vec4 before call
  any function
 */

/*!
 * @brief helper for getting sphere radius
 *
 * @param[in]   s  sphere
 *
 * @return returns radii
 */
CGLM_INLINE
float
glm_sphere_radii(vec4 s) {
  return s[3];
}

/*!
 * @brief apply transform to sphere, it is just wrapper for glm_mat4_mulv3
 *
 * @param[in]  s    sphere
 * @param[in]  m    transform matrix
 * @param[out] dest transformed sphere
 */
CGLM_INLINE
void
glm_sphere_transform(vec4 s, mat4 m, vec4 dest) {
  glm_mat4_mulv3(m, s, 1.0f, dest);
  dest[3] = s[3];
}

/*!
 * @brief merges two spheres and creates a new one
 *
 * two sphere must be in same space, for instance if one in world space then
 * the other must be in world space too, not in local space.
 *
 * @param[in]  s1   sphere 1
 * @param[in]  s2   sphere 2
 * @param[out] dest merged/extended sphere
 */
CGLM_INLINE
void
glm_sphere_merge(vec4 s1, vec4 s2, vec4 dest) {
  float dist, radii;

  dist  = glm_vec3_distance(s1, s2);
  radii = dist + s1[3] + s2[3];

  radii = glm_max(radii, s1[3]);
  radii = glm_max(radii, s2[3]);

  glm_vec3_center(s1, s2, dest);
  dest[3] = radii;
}

/*!
 * @brief check if two sphere intersects
 *
 * @param[in]   s1  sphere
 * @param[in]   s2  other sphere
 */
CGLM_INLINE
bool
glm_sphere_sphere(vec4 s1, vec4 s2) {
  return glm_vec3_distance2(s1, s2) <= glm_pow2(s1[3] + s2[3]);
}

/*!
 * @brief check if sphere intersects with point
 *
 * @param[in]   s      sphere
 * @param[in]   point  point
 */
CGLM_INLINE
bool
glm_sphere_point(vec4 s, vec3 point) {
  float rr;
  rr = s[3] * s[3];
  return glm_vec3_distance2(point, s) <= rr;
}

#endif /* cglm_sphere_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_ease_h
#define cglm_ease_h


CGLM_INLINE
float
glm_ease_linear(float t) {
  return t;
}

CGLM_INLINE
float
glm_ease_sine_in(float t) {
  return sinf((t - 1.0f) * GLM_PI_2f) + 1.0f;
}

CGLM_INLINE
float
glm_ease_sine_out(float t) {
  return sinf(t * GLM_PI_2f);
}

CGLM_INLINE
float
glm_ease_sine_inout(float t) {
  return 0.5f * (1.0f - cosf(t * GLM_PIf));
}

CGLM_INLINE
float
glm_ease_quad_in(float t) {
  return t * t;
}

CGLM_INLINE
float
glm_ease_quad_out(float t) {
  return -(t * (t - 2.0f));
}

CGLM_INLINE
float
glm_ease_quad_inout(float t) {
  float tt;

  tt = t * t;
  if (t < 0.5f)
    return 2.0f * tt;

  return (-2.0f * tt) + (4.0f * t) - 1.0f;
}

CGLM_INLINE
float
glm_ease_cubic_in(float t) {
  return t * t * t;
}

CGLM_INLINE
float
glm_ease_cubic_out(float t) {
  float f;
  f = t - 1.0f;
  return f * f * f + 1.0f;
}

CGLM_INLINE
float
glm_ease_cubic_inout(float t) {
  float f;

  if (t < 0.5f)
    return 4.0f * t * t * t;

  f = 2.0f * t - 2.0f;

  return 0.5f * f * f * f + 1.0f;
}

CGLM_INLINE
float
glm_ease_quart_in(float t) {
  float f;
  f = t * t;
  return f * f;
}

CGLM_INLINE
float
glm_ease_quart_out(float t) {
  float f;

  f = t - 1.0f;

  return f * f * f * (1.0f - t) + 1.0f;
}

CGLM_INLINE
float
glm_ease_quart_inout(float t) {
  float f, g;

  if (t < 0.5f) {
    f = t * t;
    return 8.0f * f * f;
  }

  f = t - 1.0f;
  g = f * f;

  return -8.0f * g * g + 1.0f;
}

CGLM_INLINE
float
glm_ease_quint_in(float t) {
  float f;
  f = t * t;
  return f * f * t;
}

CGLM_INLINE
float
glm_ease_quint_out(float t) {
  float f, g;

  f = t - 1.0f;
  g = f * f;

  return g * g * f + 1.0f;
}

CGLM_INLINE
float
glm_ease_quint_inout(float t) {
  float f, g;

  if (t < 0.5f) {
    f = t * t;
    return 16.0f * f * f * t;
  }

  f = 2.0f * t - 2.0f;
  g = f * f;

  return 0.5f * g * g * f + 1.0f;
}

CGLM_INLINE
float
glm_ease_exp_in(float t) {
  if (t == 0.0f)
    return t;

  return powf(2.0f,  10.0f * (t - 1.0f));
}

CGLM_INLINE
float
glm_ease_exp_out(float t) {
  if (t == 1.0f)
    return t;

  return 1.0f - powf(2.0f, -10.0f * t);
}

CGLM_INLINE
float
glm_ease_exp_inout(float t) {
  if (t == 0.0f || t == 1.0f)
    return t;

  if (t < 0.5f)
    return 0.5f * powf(2.0f, (20.0f * t) - 10.0f);

  return -0.5f * powf(2.0f, (-20.0f * t) + 10.0f) + 1.0f;
}

CGLM_INLINE
float
glm_ease_circ_in(float t) {
  return 1.0f - sqrtf(1.0f - (t * t));
}

CGLM_INLINE
float
glm_ease_circ_out(float t) {
  return sqrtf((2.0f - t) * t);
}

CGLM_INLINE
float
glm_ease_circ_inout(float t) {
  if (t < 0.5f)
    return 0.5f * (1.0f - sqrtf(1.0f - 4.0f * (t * t)));

  return 0.5f * (sqrtf(-((2.0f * t) - 3.0f) * ((2.0f * t) - 1.0f)) + 1.0f);
}

CGLM_INLINE
float
glm_ease_back_in(float t) {
  float o, z;

  o = 1.70158f;
  z = ((o + 1.0f) * t) - o;

  return t * t * z;
}

CGLM_INLINE
float
glm_ease_back_out(float t) {
  float o, z, n;

  o = 1.70158f;
  n = t - 1.0f;
  z = (o + 1.0f) * n + o;

  return n * n * z + 1.0f;
}

CGLM_INLINE
float
glm_ease_back_inout(float t) {
  float o, z, n, m, s, x;

  o = 1.70158f;
  s = o * 1.525f;
  x = 0.5;
  n = t / 0.5f;

  if (n < 1.0f) {
    z = (s + 1) * n - s;
    m = n * n * z;
    return x * m;
  }

  n -= 2.0f;
  z  = (s + 1.0f) * n + s;
  m  = (n * n * z) + 2;

  return x * m;
}

CGLM_INLINE
float
glm_ease_elast_in(float t) {
  return sinf(13.0f * GLM_PI_2f * t) * powf(2.0f, 10.0f * (t - 1.0f));
}

CGLM_INLINE
float
glm_ease_elast_out(float t) {
  return sinf(-13.0f * GLM_PI_2f * (t + 1.0f)) * powf(2.0f, -10.0f * t) + 1.0f;
}

CGLM_INLINE
float
glm_ease_elast_inout(float t) {
  float a;

  a = 2.0f * t;

  if (t < 0.5f)
    return 0.5f * sinf(13.0f * GLM_PI_2f * a)
                * powf(2.0f, 10.0f * (a - 1.0f));

  return 0.5f * (sinf(-13.0f * GLM_PI_2f * a)
                 * powf(2.0f, -10.0f * (a - 1.0f)) + 2.0f);
}

CGLM_INLINE
float
glm_ease_bounce_out(float t) {
  float tt;

  tt = t * t;

  if (t < (4.0f / 11.0f))
    return (121.0f * tt) / 16.0f;

  if (t < 8.0f / 11.0f)
    return ((363.0f / 40.0f) * tt) - ((99.0f / 10.0f) * t) + (17.0f / 5.0f);

  if (t < (9.0f / 10.0f))
    return (4356.0f / 361.0f) * tt
            - (35442.0f / 1805.0f) * t
            + (16061.0f / 1805.0f);

  return ((54.0f / 5.0f) * tt) - ((513.0f / 25.0f) * t) + (268.0f / 25.0f);
}

CGLM_INLINE
float
glm_ease_bounce_in(float t) {
  return 1.0f - glm_ease_bounce_out(1.0f - t);
}

CGLM_INLINE
float
glm_ease_bounce_inout(float t) {
  if (t < 0.5f)
    return 0.5f * (1.0f - glm_ease_bounce_out(t * 2.0f));

  return 0.5f * glm_ease_bounce_out(t * 2.0f - 1.0f) + 0.5f;
}

#endif /* cglm_ease_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_curve_h
#define cglm_curve_h


/*!
 * @brief helper function to calculate S*M*C multiplication for curves
 *
 * This function does not encourage you to use SMC,
 * instead it is a helper if you use SMC.
 *
 * if you want to specify S as vector then use more generic glm_mat4_rmc() func.
 *
 * Example usage:
 *  B(s) = glm_smc(s, GLM_BEZIER_MAT, (vec4){p0, c0, c1, p1})
 *
 * @param[in]  s  parameter between 0 and 1 (this will be [s3, s2, s, 1])
 * @param[in]  m  basis matrix
 * @param[in]  c  position/control vector
 *
 * @return B(s)
 */
CGLM_INLINE
float
glm_smc(float s, mat4 m, vec4 c) {
  vec4 vs;
  glm_vec4_cubic(s, vs);
  return glm_mat4_rmc(vs, m, c);
}

#endif /* cglm_curve_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_bezier_h
#define cglm_bezier_h


#define GLM_BEZIER_MAT_INIT  {{-1.0f,  3.0f, -3.0f,  1.0f},                   \
                              { 3.0f, -6.0f,  3.0f,  0.0f},                   \
                              {-3.0f,  3.0f,  0.0f,  0.0f},                   \
                              { 1.0f,  0.0f,  0.0f,  0.0f}}
#define GLM_HERMITE_MAT_INIT {{ 2.0f, -3.0f,  0.0f,  1.0f},                   \
                              {-2.0f,  3.0f,  0.0f,  0.0f},                   \
                              { 1.0f, -2.0f,  1.0f,  0.0f},                   \
                              { 1.0f, -1.0f,  0.0f,  0.0f}}
/* for C only */
#define GLM_BEZIER_MAT  ((mat4)GLM_BEZIER_MAT_INIT)
#define GLM_HERMITE_MAT ((mat4)GLM_HERMITE_MAT_INIT)

#define CGLM_DECASTEL_EPS   1e-9f
#define CGLM_DECASTEL_MAX   1000.0f
#define CGLM_DECASTEL_SMALL 1e-20f

/*!
 * @brief cubic bezier interpolation
 *
 * Formula:
 *  B(s) = P0*(1-s)^3 + 3*C0*s*(1-s)^2 + 3*C1*s^2*(1-s) + P1*s^3
 *
 * similar result using matrix:
 *  B(s) = glm_smc(t, GLM_BEZIER_MAT, (vec4){p0, c0, c1, p1})
 *
 * glm_eq(glm_smc(...), glm_bezier(...)) should return TRUE
 *
 * @param[in]  s    parameter between 0 and 1
 * @param[in]  p0   begin point
 * @param[in]  c0   control point 1
 * @param[in]  c1   control point 2
 * @param[in]  p1   end point
 *
 * @return B(s)
 */
CGLM_INLINE
float
glm_bezier(float s, float p0, float c0, float c1, float p1) {
  float x, xx, ss, xs3, a;

  x   = 1.0f - s;
  xx  = x * x;
  ss  = s * s;
  xs3 = (s - ss) * 3.0f;
  a   = p0 * xx + c0 * xs3;

  return a + s * (c1 * xs3 + p1 * ss - a);
}

/*!
 * @brief cubic hermite interpolation
 *
 * Formula:
 *  H(s) = P0*(2*s^3 - 3*s^2 + 1) + T0*(s^3 - 2*s^2 + s)
 *            + P1*(-2*s^3 + 3*s^2) + T1*(s^3 - s^2)
 *
 * similar result using matrix:
 *  H(s) = glm_smc(t, GLM_HERMITE_MAT, (vec4){p0, p1, c0, c1})
 *
 * glm_eq(glm_smc(...), glm_hermite(...)) should return TRUE
 *
 * @param[in]  s    parameter between 0 and 1
 * @param[in]  p0   begin point
 * @param[in]  t0   tangent 1
 * @param[in]  t1   tangent 2
 * @param[in]  p1   end point
 *
 * @return H(s)
 */
CGLM_INLINE
float
glm_hermite(float s, float p0, float t0, float t1, float p1) {
  float ss, d, a, b, c, e, f;

  ss = s  * s;
  a  = ss + ss;
  c  = a  + ss;
  b  = a  * s;
  d  = s  * ss;
  f  = d  - ss;
  e  = b  - c;

  return p0 * (e + 1.0f) + t0 * (f - ss + s) + t1 * f - p1 * e;
}

/*!
 * @brief iterative way to solve cubic equation
 *
 * @param[in]  prm  parameter between 0 and 1
 * @param[in]  p0   begin point
 * @param[in]  c0   control point 1
 * @param[in]  c1   control point 2
 * @param[in]  p1   end point
 *
 * @return parameter to use in cubic equation
 */
CGLM_INLINE
float
glm_decasteljau(float prm, float p0, float c0, float c1, float p1) {
  float u, v, a, b, c, d, e, f;
  int   i;

  if (prm - p0 < CGLM_DECASTEL_SMALL)
    return 0.0f;

  if (p1 - prm < CGLM_DECASTEL_SMALL)
    return 1.0f;

  u  = 0.0f;
  v  = 1.0f;

  for (i = 0; i < CGLM_DECASTEL_MAX; i++) {
    /* de Casteljau Subdivision */
    a  = (p0 + c0) * 0.5f;
    b  = (c0 + c1) * 0.5f;
    c  = (c1 + p1) * 0.5f;
    d  = (a  + b)  * 0.5f;
    e  = (b  + c)  * 0.5f;
    f  = (d  + e)  * 0.5f; /* this one is on the curve! */

    /* The curve point is close enough to our wanted t */
    if (fabsf(f - prm) < CGLM_DECASTEL_EPS)
      return glm_clamp_zo((u  + v) * 0.5f);

    /* dichotomy */
    if (f < prm) {
      p0 = f;
      c0 = e;
      c1 = c;
      u  = (u  + v) * 0.5f;
    } else {
      c0 = a;
      c1 = d;
      p1 = f;
      v  = (u  + v) * 0.5f;
    }
  }

  return glm_clamp_zo((u  + v) * 0.5f);
}

#endif /* cglm_bezier_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*
 Functions:
   CGLM_INLINE bool glm_line_triangle_intersect(vec3   origin,
                                                vec3   direction,
                                                vec3   v0,
                                                vec3   v1,
                                                vec3   v2,
                                                float *d);
*/

#ifndef cglm_ray_h
#define cglm_ray_h


/*!
 * @brief Möller–Trumbore ray-triangle intersection algorithm
 * 
 * @param[in] origin         origin of ray
 * @param[in] direction      direction of ray
 * @param[in] v0             first vertex of triangle
 * @param[in] v1             second vertex of triangle
 * @param[in] v2             third vertex of triangle
 * @param[in, out] d         distance to intersection
 * @return whether there is intersection
 */

CGLM_INLINE
bool
glm_ray_triangle(vec3   origin,
                 vec3   direction,
                 vec3   v0,
                 vec3   v1,
                 vec3   v2,
                 float *d) {
  vec3        edge1, edge2, p, t, q;
  float       det, inv_det, u, v, dist;
  const float epsilon = 0.000001f;

  glm_vec3_sub(v1, v0, edge1);
  glm_vec3_sub(v2, v0, edge2);
  glm_vec3_cross(direction, edge2, p);

  det = glm_vec3_dot(edge1, p);
  if (det > -epsilon && det < epsilon)
    return false;

  inv_det = 1.0f / det;
  
  glm_vec3_sub(origin, v0, t);

  u = inv_det * glm_vec3_dot(t, p);
  if (u < 0.0f || u > 1.0f)
    return false;

  glm_vec3_cross(t, edge1, q);

  v = inv_det * glm_vec3_dot(direction, q);
  if (v < 0.0f || u + v > 1.0f)
    return false;

  dist = inv_det * glm_vec3_dot(edge2, q);

  if (d)
    *d = dist;

  return dist > epsilon;
}

#endif

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

/*
 Functions:
   CGLM_INLINE void glm_translate2d(mat3 m, vec2 v)
   CGLM_INLINE void glm_translate2d_to(mat3 m, vec2 v, mat3 dest)
   CGLM_INLINE void glm_translate2d_x(mat3 m, float x)
   CGLM_INLINE void glm_translate2d_y(mat3 m, float y)
   CGLM_INLINE void glm_translate2d_make(mat3 m, vec2 v)
   CGLM_INLINE void glm_scale2d_to(mat3 m, vec2 v, mat3 dest)
   CGLM_INLINE void glm_scale2d_make(mat3 m, vec2 v)
   CGLM_INLINE void glm_scale2d(mat3 m, vec2 v)
   CGLM_INLINE void glm_scale2d_uni(mat3 m, float s)
   CGLM_INLINE void glm_rotate2d_make(mat3 m, float angle)
   CGLM_INLINE void glm_rotate2d(mat3 m, float angle)
   CGLM_INLINE void glm_rotate2d_to(mat3 m, float angle, mat3 dest)
 */

#ifndef cglm_affine2d_h
#define cglm_affine2d_h


/*!
 * @brief translate existing 2d transform matrix by v vector
 *        and stores result in same matrix
 *
 * @param[in, out]  m  affine transfrom
 * @param[in]       v  translate vector [x, y]
 */
CGLM_INLINE
void
glm_translate2d(mat3 m, vec2 v) {
  m[2][0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0];
  m[2][1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1];
  m[2][2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2];
}

/*!
 * @brief translate existing 2d transform matrix by v vector
 *        and store result in dest
 *
 * source matrix will remain same
 *
 * @param[in]  m    affine transfrom
 * @param[in]  v    translate vector [x, y]
 * @param[out] dest translated matrix
 */
CGLM_INLINE
void
glm_translate2d_to(mat3 m, vec2 v, mat3 dest) {
  glm_mat3_copy(m, dest);
  glm_translate2d(dest, v);
}

/*!
 * @brief translate existing 2d transform matrix by x factor
 *
 * @param[in, out]  m  affine transfrom
 * @param[in]       x  x factor
 */
CGLM_INLINE
void
glm_translate2d_x(mat3 m, float x) {
  m[2][0] = m[0][0] * x + m[2][0];
  m[2][1] = m[0][1] * x + m[2][1];
  m[2][2] = m[0][2] * x + m[2][2];
}

/*!
 * @brief translate existing 2d transform matrix by y factor
 *
 * @param[in, out]  m  affine transfrom
 * @param[in]       y  y factor
 */
CGLM_INLINE
void
glm_translate2d_y(mat3 m, float y) {
  m[2][0] = m[1][0] * y + m[2][0];
  m[2][1] = m[1][1] * y + m[2][1];
  m[2][2] = m[1][2] * y + m[2][2];
}

/*!
 * @brief creates NEW translate 2d transform matrix by v vector
 *
 * @param[out]  m  affine transfrom
 * @param[in]   v  translate vector [x, y]
 */
CGLM_INLINE
void
glm_translate2d_make(mat3 m, vec2 v) {
  glm_mat3_identity(m);
  m[2][0] = v[0];
  m[2][1] = v[1];
}

/*!
 * @brief scale existing 2d transform matrix by v vector
 *        and store result in dest
 *
 * @param[in]  m    affine transfrom
 * @param[in]  v    scale vector [x, y]
 * @param[out] dest scaled matrix
 */
CGLM_INLINE
void
glm_scale2d_to(mat3 m, vec2 v, mat3 dest) {
  dest[0][0] = m[0][0] * v[0];
  dest[0][1] = m[0][1] * v[0];
  dest[0][2] = m[0][2] * v[0];
  
  dest[1][0] = m[1][0] * v[1];
  dest[1][1] = m[1][1] * v[1];
  dest[1][2] = m[1][2] * v[1];
  
  dest[2][0] = m[2][0];
  dest[2][1] = m[2][1];
  dest[2][2] = m[2][2];
}

/*!
 * @brief creates NEW 2d scale matrix by v vector
 *
 * @param[out]  m  affine transfrom
 * @param[in]   v  scale vector [x, y]
 */
CGLM_INLINE
void
glm_scale2d_make(mat3 m, vec2 v) {
  glm_mat3_identity(m);
  m[0][0] = v[0];
  m[1][1] = v[1];
}

/*!
 * @brief scales existing 2d transform matrix by v vector
 *        and stores result in same matrix
 *
 * @param[in, out]  m  affine transfrom
 * @param[in]       v  scale vector [x, y]
 */
CGLM_INLINE
void
glm_scale2d(mat3 m, vec2 v) {
  m[0][0] = m[0][0] * v[0];
  m[0][1] = m[0][1] * v[0];
  m[0][2] = m[0][2] * v[0];

  m[1][0] = m[1][0] * v[1];
  m[1][1] = m[1][1] * v[1];
  m[1][2] = m[1][2] * v[1];
}

/*!
 * @brief applies uniform scale to existing 2d transform matrix v = [s, s]
 *        and stores result in same matrix
 *
 * @param[in, out]  m  affine transfrom
 * @param[in]       s  scale factor
 */
CGLM_INLINE
void
glm_scale2d_uni(mat3 m, float s) {
  m[0][0] = m[0][0] * s;
  m[0][1] = m[0][1] * s;
  m[0][2] = m[0][2] * s;

  m[1][0] = m[1][0] * s;
  m[1][1] = m[1][1] * s;
  m[1][2] = m[1][2] * s;
}

/*!
 * @brief creates NEW rotation matrix by angle around Z axis
 *
 * @param[out] m     affine transfrom
 * @param[in]  angle angle (radians)
 */
CGLM_INLINE
void
glm_rotate2d_make(mat3 m, float angle) {
  float c, s;

  s = sinf(angle);
  c = cosf(angle);
  
  m[0][0] = c;
  m[0][1] = s;
  m[0][2] = 0;

  m[1][0] = -s;
  m[1][1] = c;
  m[1][2] = 0;
  
  m[2][0] = 0.0f;
  m[2][1] = 0.0f;
  m[2][2] = 1.0f;
}

/*!
 * @brief rotate existing 2d transform matrix around Z axis by angle
 *         and store result in same matrix
 *
 * @param[in, out]  m      affine transfrom
 * @param[in]       angle  angle (radians)
 */
CGLM_INLINE
void
glm_rotate2d(mat3 m, float angle) {
  float m00 = m[0][0],  m10 = m[1][0],
        m01 = m[0][1],  m11 = m[1][1],
        m02 = m[0][2],  m12 = m[1][2];
  float c, s;

  s = sinf(angle);
  c = cosf(angle);
  
  m[0][0] = m00 * c + m10 * s;
  m[0][1] = m01 * c + m11 * s;
  m[0][2] = m02 * c + m12 * s;

  m[1][0] = m00 * -s + m10 * c;
  m[1][1] = m01 * -s + m11 * c;
  m[1][2] = m02 * -s + m12 * c;
}

/*!
 * @brief rotate existing 2d transform matrix around Z axis by angle
 *        and store result in dest
 *
 * @param[in]  m      affine transfrom
 * @param[in]  angle  angle (radians)
 * @param[out] dest   destination
 */
CGLM_INLINE
void
glm_rotate2d_to(mat3 m, float angle, mat3 dest) {
  float m00 = m[0][0],  m10 = m[1][0],
        m01 = m[0][1],  m11 = m[1][1],
        m02 = m[0][2],  m12 = m[1][2];
  float c, s;

  s = sinf(angle);
  c = cosf(angle);
  
  dest[0][0] = m00 * c + m10 * s;
  dest[0][1] = m01 * c + m11 * s;
  dest[0][2] = m02 * c + m12 * s;

  dest[1][0] = m00 * -s + m10 * c;
  dest[1][1] = m01 * -s + m11 * c;
  dest[1][2] = m02 * -s + m12 * c;
  
  dest[2][0] = m[2][0];
  dest[2][1] = m[2][1];
  dest[2][2] = m[2][2];
}

#endif /* cglm_affine2d_h */


#endif /* cglm_h */


#ifdef __cplusplus
}
#endif

