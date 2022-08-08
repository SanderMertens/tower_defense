/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#include "cglm.h"
/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglm_call_h
#define cglm_call_h
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_vec2_h
#define cglmc_vec2_h
#ifdef __cplusplus
extern "C" {
#endif


CGLM_EXPORT
void
glmc_vec2(float * __restrict v, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_copy(vec2 a, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_zero(vec2 v);

CGLM_EXPORT
void
glmc_vec2_one(vec2 v);

CGLM_EXPORT
float
glmc_vec2_dot(vec2 a, vec2 b);

CGLM_EXPORT
float
glmc_vec2_cross(vec2 a, vec2 b);

CGLM_EXPORT
float
glmc_vec2_norm2(vec2 v);

CGLM_EXPORT
float
glmc_vec2_norm(vec2 v);

CGLM_EXPORT
void
glmc_vec2_add(vec2 a, vec2 b, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_adds(vec2 v, float s, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_sub(vec2 a, vec2 b, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_subs(vec2 v, float s, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_mul(vec2 a, vec2 b, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_scale(vec2 v, float s, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_scale_as(vec2 v, float s, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_div(vec2 a, vec2 b, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_divs(vec2 v, float s, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_addadd(vec2 a, vec2 b, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_subadd(vec2 a, vec2 b, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_muladd(vec2 a, vec2 b, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_muladds(vec2 a, float s, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_maxadd(vec2 a, vec2 b, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_minadd(vec2 a, vec2 b, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_negate_to(vec2 v, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_negate(vec2 v);

CGLM_EXPORT
void
glmc_vec2_normalize(vec2 v);

CGLM_EXPORT
void
glmc_vec2_normalize_to(vec2 v, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_rotate(vec2 v, float angle, vec2 dest);

CGLM_EXPORT
float
glmc_vec2_distance2(vec2 a, vec2 b);

CGLM_EXPORT
float
glmc_vec2_distance(vec2 a, vec2 b);

CGLM_EXPORT
void
glmc_vec2_maxv(vec2 a, vec2 b, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_minv(vec2 a, vec2 b, vec2 dest);

CGLM_EXPORT
void
glmc_vec2_clamp(vec2 v, float minval, float maxval);

CGLM_EXPORT
void
glmc_vec2_lerp(vec2 from, vec2 to, float t, vec2 dest);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_vec2_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_vec3_h
#define cglmc_vec3_h
#ifdef __cplusplus
extern "C" {
#endif


/* DEPRECATED! use _copy, _ucopy versions */
#define glmc_vec_dup(v, dest)          glmc_vec3_copy(v, dest)
#define glmc_vec3_flipsign(v)          glmc_vec3_negate(v)
#define glmc_vec3_flipsign_to(v, dest) glmc_vec3_negate_to(v, dest)
#define glmc_vec3_inv(v)               glmc_vec3_negate(v)
#define glmc_vec3_inv_to(v, dest)      glmc_vec3_negate_to(v, dest)

CGLM_EXPORT
void
glmc_vec3(vec4 v4, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_copy(vec3 a, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_zero(vec3 v);

CGLM_EXPORT
void
glmc_vec3_one(vec3 v);

CGLM_EXPORT
float
glmc_vec3_dot(vec3 a, vec3 b);

CGLM_EXPORT
void
glmc_vec3_cross(vec3 a, vec3 b, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_crossn(vec3 a, vec3 b, vec3 dest);

CGLM_EXPORT
float
glmc_vec3_norm(vec3 v);

CGLM_EXPORT
float
glmc_vec3_norm2(vec3 v);
    
CGLM_EXPORT
float
glmc_vec3_norm_one(vec3 v);

CGLM_EXPORT
float
glmc_vec3_norm_inf(vec3 v);

CGLM_EXPORT
void
glmc_vec3_normalize_to(vec3 v, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_normalize(vec3 v);

CGLM_EXPORT
void
glmc_vec3_add(vec3 a, vec3 b, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_adds(vec3 v, float s, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_sub(vec3 a, vec3 b, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_subs(vec3 v, float s, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_mul(vec3 a, vec3 b, vec3 d);

CGLM_EXPORT
void
glmc_vec3_scale(vec3 v, float s, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_scale_as(vec3 v, float s, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_div(vec3 a, vec3 b, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_divs(vec3 a, float s, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_addadd(vec3 a, vec3 b, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_subadd(vec3 a, vec3 b, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_muladd(vec3 a, vec3 b, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_muladds(vec3 a, float s, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_maxadd(vec3 a, vec3 b, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_minadd(vec3 a, vec3 b, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_negate(vec3 v);

CGLM_EXPORT
void
glmc_vec3_negate_to(vec3 v, vec3 dest);

CGLM_EXPORT
float
glmc_vec3_angle(vec3 a, vec3 b);

CGLM_EXPORT
void
glmc_vec3_rotate(vec3 v, float angle, vec3 axis);

CGLM_EXPORT
void
glmc_vec3_rotate_m4(mat4 m, vec3 v, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_rotate_m3(mat3 m, vec3 v, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_proj(vec3 a, vec3 b, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_center(vec3 a, vec3 b, vec3 dest);

CGLM_EXPORT
float
glmc_vec3_distance2(vec3 a, vec3 b);

CGLM_EXPORT
float
glmc_vec3_distance(vec3 a, vec3 b);

CGLM_EXPORT
void
glmc_vec3_maxv(vec3 a, vec3 b, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_minv(vec3 a, vec3 b, vec3 dest);

CGLM_EXPORT
void
glmc_vec3_clamp(vec3 v, float minVal, float maxVal);

CGLM_EXPORT
void
glmc_vec3_ortho(vec3 v, vec3 dest);
    
CGLM_EXPORT
void
glmc_vec3_lerp(vec3 from, vec3 to, float t, vec3 dest);
    
CGLM_EXPORT
void
glmc_vec3_lerpc(vec3 from, vec3 to, float t, vec3 dest);
    
CGLM_INLINE
void
glmc_vec3_mix(vec3 from, vec3 to, float t, vec3 dest) {
  glmc_vec3_lerp(from, to, t, dest);
}

CGLM_INLINE
void
glmc_vec3_mixc(vec3 from, vec3 to, float t, vec3 dest) {
  glmc_vec3_lerpc(from, to, t, dest);
}
    
CGLM_EXPORT
void
glmc_vec3_step_uni(float edge, vec3 x, vec3 dest);
    
CGLM_EXPORT
void
glmc_vec3_step(vec3 edge, vec3 x, vec3 dest);
    
CGLM_EXPORT
void
glmc_vec3_smoothstep_uni(float edge0, float edge1, vec3 x, vec3 dest);
    
CGLM_EXPORT
void
glmc_vec3_smoothstep(vec3 edge0, vec3 edge1, vec3 x, vec3 dest);
    
CGLM_EXPORT
void
glmc_vec3_smoothinterp(vec3 from, vec3 to, float t, vec3 dest);
    
CGLM_EXPORT
void
glmc_vec3_smoothinterpc(vec3 from, vec3 to, float t, vec3 dest);

/* ext */

CGLM_EXPORT
void
glmc_vec3_mulv(vec3 a, vec3 b, vec3 d);

CGLM_EXPORT
void
glmc_vec3_broadcast(float val, vec3 d);
    
CGLM_EXPORT
void
glmc_vec3_fill(vec3 v, float val);

CGLM_EXPORT
bool
glmc_vec3_eq(vec3 v, float val);

CGLM_EXPORT
bool
glmc_vec3_eq_eps(vec3 v, float val);

CGLM_EXPORT
bool
glmc_vec3_eq_all(vec3 v);

CGLM_EXPORT
bool
glmc_vec3_eqv(vec3 a, vec3 b);

CGLM_EXPORT
bool
glmc_vec3_eqv_eps(vec3 a, vec3 b);

CGLM_EXPORT
float
glmc_vec3_max(vec3 v);

CGLM_EXPORT
float
glmc_vec3_min(vec3 v);

CGLM_EXPORT
bool
glmc_vec3_isnan(vec3 v);

CGLM_EXPORT
bool
glmc_vec3_isinf(vec3 v);

CGLM_EXPORT
bool
glmc_vec3_isvalid(vec3 v);

CGLM_EXPORT
void
glmc_vec3_sign(vec3 v, vec3 dest);
    
CGLM_EXPORT
void
glmc_vec3_abs(vec3 v, vec3 dest);
    
CGLM_EXPORT
void
glmc_vec3_fract(vec3 v, vec3 dest);
    
CGLM_EXPORT
float
glmc_vec3_hadd(vec3 v);

CGLM_EXPORT
void
glmc_vec3_sqrt(vec3 v, vec3 dest);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_vec3_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_vec4_h
#define cglmc_vec4_h
#ifdef __cplusplus
extern "C" {
#endif


/* DEPRECATED! use _copy, _ucopy versions */
#define glmc_vec4_dup3(v, dest)         glmc_vec4_copy3(v, dest)
#define glmc_vec4_dup(v, dest)          glmc_vec4_copy(v, dest)
#define glmc_vec4_flipsign(v)           glmc_vec4_negate(v)
#define glmc_vec4_flipsign_to(v, dest)  glmc_vec4_negate_to(v, dest)
#define glmc_vec4_inv(v)                glmc_vec4_negate(v)
#define glmc_vec4_inv_to(v, dest)       glmc_vec4_negate_to(v, dest)

CGLM_EXPORT
void
glmc_vec4(vec3 v3, float last, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_zero(vec4 v);

CGLM_EXPORT
void
glmc_vec4_one(vec4 v);

CGLM_EXPORT
void
glmc_vec4_copy3(vec4 v, vec3 dest);

CGLM_EXPORT
void
glmc_vec4_copy(vec4 v, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_ucopy(vec4 v, vec4 dest);

CGLM_EXPORT
float
glmc_vec4_dot(vec4 a, vec4 b);

CGLM_EXPORT
float
glmc_vec4_norm(vec4 v);

CGLM_EXPORT
float
glmc_vec4_norm2(vec4 v);
    
CGLM_EXPORT
float
glmc_vec4_norm_one(vec4 v);

CGLM_EXPORT
float
glmc_vec4_norm_inf(vec4 v);

CGLM_EXPORT
void
glmc_vec4_normalize_to(vec4 v, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_normalize(vec4 v);

CGLM_EXPORT
void
glmc_vec4_add(vec4 a, vec4 b, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_adds(vec4 v, float s, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_sub(vec4 a, vec4 b, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_subs(vec4 v, float s, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_mul(vec4 a, vec4 b, vec4 d);

CGLM_EXPORT
void
glmc_vec4_scale(vec4 v, float s, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_scale_as(vec3 v, float s, vec3 dest);

CGLM_EXPORT
void
glmc_vec4_div(vec4 a, vec4 b, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_divs(vec4 v, float s, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_addadd(vec4 a, vec4 b, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_subadd(vec4 a, vec4 b, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_muladd(vec4 a, vec4 b, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_muladds(vec4 a, float s, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_maxadd(vec4 a, vec4 b, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_minadd(vec4 a, vec4 b, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_negate(vec4 v);

CGLM_EXPORT
void
glmc_vec4_negate_to(vec4 v, vec4 dest);
    
CGLM_EXPORT
float
glmc_vec4_distance(vec4 a, vec4 b);
    
CGLM_EXPORT
float
glmc_vec4_distance2(vec4 a, vec4 b);

CGLM_EXPORT
void
glmc_vec4_maxv(vec4 a, vec4 b, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_minv(vec4 a, vec4 b, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_clamp(vec4 v, float minVal, float maxVal);
    
CGLM_EXPORT
void
glmc_vec4_lerp(vec4 from, vec4 to, float t, vec4 dest);
    
CGLM_EXPORT
void
glmc_vec4_lerpc(vec4 from, vec4 to, float t, vec4 dest);
    
CGLM_INLINE
void
glmc_vec4_mix(vec4 from, vec4 to, float t, vec4 dest) {
  glmc_vec4_lerp(from, to, t, dest);
}

CGLM_INLINE
void
glmc_vec4_mixc(vec4 from, vec4 to, float t, vec4 dest) {
  glmc_vec4_lerpc(from, to, t, dest);
}
    
CGLM_EXPORT
void
glmc_vec4_step_uni(float edge, vec4 x, vec4 dest);
    
CGLM_EXPORT
void
glmc_vec4_step(vec4 edge, vec4 x, vec4 dest);
    
CGLM_EXPORT
void
glmc_vec4_smoothstep_uni(float edge0, float edge1, vec4 x, vec4 dest);
    
CGLM_EXPORT
void
glmc_vec4_smoothstep(vec4 edge0, vec4 edge1, vec4 x, vec4 dest);
    
CGLM_EXPORT
void
glmc_vec4_smoothinterp(vec4 from, vec4 to, float t, vec4 dest);
    
CGLM_EXPORT
void
glmc_vec4_smoothinterpc(vec4 from, vec4 to, float t, vec4 dest);

CGLM_EXPORT
void
glmc_vec4_cubic(float s, vec4 dest);

/* ext */

CGLM_EXPORT
void
glmc_vec4_mulv(vec4 a, vec4 b, vec4 d);

CGLM_EXPORT
void
glmc_vec4_broadcast(float val, vec4 d);
    
CGLM_EXPORT
void
glmc_vec4_fill(vec4 v, float val);

CGLM_EXPORT
bool
glmc_vec4_eq(vec4 v, float val);

CGLM_EXPORT
bool
glmc_vec4_eq_eps(vec4 v, float val);

CGLM_EXPORT
bool
glmc_vec4_eq_all(vec4 v);

CGLM_EXPORT
bool
glmc_vec4_eqv(vec4 a, vec4 b);

CGLM_EXPORT
bool
glmc_vec4_eqv_eps(vec4 a, vec4 b);

CGLM_EXPORT
float
glmc_vec4_max(vec4 v);

CGLM_EXPORT
float
glmc_vec4_min(vec4 v);

CGLM_EXPORT
bool
glmc_vec4_isnan(vec4 v);

CGLM_EXPORT
bool
glmc_vec4_isinf(vec4 v);

CGLM_EXPORT
bool
glmc_vec4_isvalid(vec4 v);

CGLM_EXPORT
void
glmc_vec4_sign(vec4 v, vec4 dest);
    
CGLM_EXPORT
void
glmc_vec4_abs(vec4 v, vec4 dest);
    
CGLM_EXPORT
void
glmc_vec4_fract(vec4 v, vec4 dest);
    
CGLM_EXPORT
float
glmc_vec4_hadd(vec4 v);

CGLM_EXPORT
void
glmc_vec4_sqrt(vec4 v, vec4 dest);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_vec4_h */


/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_mat2_h
#define cglmc_mat2_h
#ifdef __cplusplus
extern "C" {
#endif


CGLM_EXPORT
void
glmc_mat2_copy(mat2 mat, mat2 dest);

CGLM_EXPORT
void
glmc_mat2_identity(mat2 mat);

CGLM_EXPORT
void
glmc_mat2_identity_array(mat2 * __restrict mat, size_t count);

CGLM_EXPORT
void
glmc_mat2_zero(mat2 mat);

CGLM_EXPORT
void
glmc_mat2_mul(mat2 m1, mat2 m2, mat2 dest);

CGLM_EXPORT
void
glmc_mat2_transpose_to(mat2 m, mat2 dest);

CGLM_EXPORT
void
glmc_mat2_transpose(mat2 m);

CGLM_EXPORT
void
glmc_mat2_mulv(mat2 m, vec2 v, vec2 dest);

CGLM_EXPORT
float
glmc_mat2_trace(mat2 m);

CGLM_EXPORT
void
glmc_mat2_scale(mat2 m, float s);

CGLM_EXPORT
float
glmc_mat2_det(mat2 mat);

CGLM_EXPORT
void
glmc_mat2_inv(mat2 mat, mat2 dest);

CGLM_EXPORT
void
glmc_mat2_swap_col(mat2 mat, int col1, int col2);

CGLM_EXPORT
void
glmc_mat2_swap_row(mat2 mat, int row1, int row2);

CGLM_EXPORT
float
glmc_mat2_rmc(vec2 r, mat2 m, vec2 c);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_mat2_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_mat3_h
#define cglmc_mat3_h
#ifdef __cplusplus
extern "C" {
#endif


/* DEPRECATED! use _copy, _ucopy versions */
#define glmc_mat3_dup(mat, dest)  glmc_mat3_copy(mat, dest)

CGLM_EXPORT
void
glmc_mat3_copy(mat3 mat, mat3 dest);

CGLM_EXPORT
void
glmc_mat3_identity(mat3 mat);

CGLM_EXPORT
void
glmc_mat3_zero(mat3 mat);

CGLM_EXPORT
void
glmc_mat3_identity_array(mat3 * __restrict mat, size_t count);

CGLM_EXPORT
void
glmc_mat3_mul(mat3 m1, mat3 m2, mat3 dest);

CGLM_EXPORT
void
glmc_mat3_transpose_to(mat3 m, mat3 dest);

CGLM_EXPORT
void
glmc_mat3_transpose(mat3 m);

CGLM_EXPORT
void
glmc_mat3_mulv(mat3 m, vec3 v, vec3 dest);

CGLM_EXPORT
float
glmc_mat3_trace(mat3 m);

CGLM_EXPORT
void
glmc_mat3_quat(mat3 m, versor dest);

CGLM_EXPORT
void
glmc_mat3_scale(mat3 m, float s);

CGLM_EXPORT
float
glmc_mat3_det(mat3 mat);

CGLM_EXPORT
void
glmc_mat3_inv(mat3 mat, mat3 dest);

CGLM_EXPORT
void
glmc_mat3_swap_col(mat3 mat, int col1, int col2);

CGLM_EXPORT
void
glmc_mat3_swap_row(mat3 mat, int row1, int row2);

CGLM_EXPORT
float
glmc_mat3_rmc(vec3 r, mat3 m, vec3 c);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_mat3_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_mat_h
#define cglmc_mat_h
#ifdef __cplusplus
extern "C" {
#endif


/* DEPRECATED! use _copy, _ucopy versions */
#define glmc_mat4_udup(mat, dest) glmc_mat4_ucopy(mat, dest)
#define glmc_mat4_dup(mat, dest)  glmc_mat4_copy(mat, dest)

CGLM_EXPORT
void
glmc_mat4_ucopy(mat4 mat, mat4 dest);

CGLM_EXPORT
void
glmc_mat4_copy(mat4 mat, mat4 dest);

CGLM_EXPORT
void
glmc_mat4_identity(mat4 mat);

CGLM_EXPORT
void
glmc_mat4_identity_array(mat4 * __restrict mat, size_t count);

CGLM_EXPORT
void
glmc_mat4_zero(mat4 mat);

CGLM_EXPORT
void
glmc_mat4_pick3(mat4 mat, mat3 dest);

CGLM_EXPORT
void
glmc_mat4_pick3t(mat4 mat, mat3 dest);

CGLM_EXPORT
void
glmc_mat4_ins3(mat3 mat, mat4 dest);

CGLM_EXPORT
void
glmc_mat4_mul(mat4 m1, mat4 m2, mat4 dest);

CGLM_EXPORT
void
glmc_mat4_mulN(mat4 * __restrict matrices[], uint32_t len, mat4 dest);

CGLM_EXPORT
void
glmc_mat4_mulv(mat4 m, vec4 v, vec4 dest);

CGLM_EXPORT
void
glmc_mat4_mulv3(mat4 m, vec3 v, float last, vec3 dest);

CGLM_EXPORT
float
glmc_mat4_trace(mat4 m);

CGLM_EXPORT
float
glmc_mat4_trace3(mat4 m);

CGLM_EXPORT
void
glmc_mat4_quat(mat4 m, versor dest);

CGLM_EXPORT
void
glmc_mat4_transpose_to(mat4 m, mat4 dest);

CGLM_EXPORT
void
glmc_mat4_transpose(mat4 m);

CGLM_EXPORT
void
glmc_mat4_scale_p(mat4 m, float s);

CGLM_EXPORT
void
glmc_mat4_scale(mat4 m, float s);

CGLM_EXPORT
float
glmc_mat4_det(mat4 mat);

CGLM_EXPORT
void
glmc_mat4_inv(mat4 mat, mat4 dest);

CGLM_EXPORT
void
glmc_mat4_inv_precise(mat4 mat, mat4 dest);

CGLM_EXPORT
void
glmc_mat4_inv_fast(mat4 mat, mat4 dest);

CGLM_EXPORT
void
glmc_mat4_swap_col(mat4 mat, int col1, int col2);

CGLM_EXPORT
void
glmc_mat4_swap_row(mat4 mat, int row1, int row2);

CGLM_EXPORT
float
glmc_mat4_rmc(vec4 r, mat4 m, vec4 c);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_mat_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_affine_h
#define cglmc_affine_h
#ifdef __cplusplus
extern "C" {
#endif


CGLM_EXPORT
void
glmc_translate_make(mat4 m, vec3 v);

CGLM_EXPORT
void
glmc_translate_to(mat4 m, vec3 v, mat4 dest);

CGLM_EXPORT
void
glmc_translate(mat4 m, vec3 v);

CGLM_EXPORT
void
glmc_translate_x(mat4 m, float to);

CGLM_EXPORT
void
glmc_translate_y(mat4 m, float to);

CGLM_EXPORT
void
glmc_translate_z(mat4 m, float to);

CGLM_EXPORT
void
glmc_scale_make(mat4 m, vec3 v);

CGLM_EXPORT
void
glmc_scale_to(mat4 m, vec3 v, mat4 dest);

CGLM_EXPORT
void
glmc_scale(mat4 m, vec3 v);

CGLM_EXPORT
void
glmc_scale_uni(mat4 m, float s);

CGLM_EXPORT
void
glmc_rotate_x(mat4 m, float rad, mat4 dest);

CGLM_EXPORT
void
glmc_rotate_y(mat4 m, float rad, mat4 dest);

CGLM_EXPORT
void
glmc_rotate_z(mat4 m, float rad, mat4 dest);

CGLM_EXPORT
void
glmc_rotate_make(mat4 m, float angle, vec3 axis);

CGLM_EXPORT
void
glmc_rotate(mat4 m, float angle, vec3 axis);

CGLM_EXPORT
void
glmc_rotate_at(mat4 m, vec3 pivot, float angle, vec3 axis);

CGLM_EXPORT
void
glmc_rotate_atm(mat4 m, vec3 pivot, float angle, vec3 axis);

CGLM_EXPORT
void
glmc_decompose_scalev(mat4 m, vec3 s);

CGLM_EXPORT
bool
glmc_uniscaled(mat4 m);

CGLM_EXPORT
void
glmc_decompose_rs(mat4 m, mat4 r, vec3 s);

CGLM_EXPORT
void
glmc_decompose(mat4 m, vec4 t, mat4 r, vec3 s);

/* affine-mat */

CGLM_EXPORT
void
glmc_mul(mat4 m1, mat4 m2, mat4 dest);

CGLM_EXPORT
void
glmc_mul_rot(mat4 m1, mat4 m2, mat4 dest);

CGLM_EXPORT
void
glmc_inv_tr(mat4 mat);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_affine_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_cam_h
#define cglmc_cam_h
#ifdef __cplusplus
extern "C" {
#endif


CGLM_EXPORT
void
glmc_frustum(float left,
             float right,
             float bottom,
             float top,
             float nearVal,
             float farVal,
             mat4 dest);

CGLM_EXPORT
void
glmc_ortho(float left,
           float right,
           float bottom,
           float top,
           float nearVal,
           float farVal,
           mat4 dest);

CGLM_EXPORT
void
glmc_ortho_aabb(vec3 box[2], mat4 dest);

CGLM_EXPORT
void
glmc_ortho_aabb_p(vec3 box[2], float padding, mat4 dest);

CGLM_EXPORT
void
glmc_ortho_aabb_pz(vec3 box[2], float padding, mat4 dest);

CGLM_EXPORT
void
glmc_ortho_default(float aspect, mat4 dest);

CGLM_EXPORT
void
glmc_ortho_default_s(float aspect, float size, mat4 dest);

CGLM_EXPORT
void
glmc_perspective(float fovy,
                 float aspect,
                 float nearVal,
                 float farVal,
                 mat4 dest);

CGLM_EXPORT
void
glmc_persp_move_far(mat4 proj, float deltaFar);

CGLM_EXPORT
void
glmc_perspective_default(float aspect, mat4 dest);

CGLM_EXPORT
void
glmc_perspective_resize(float aspect, mat4 proj);

CGLM_EXPORT
void
glmc_lookat(vec3 eye, vec3 center, vec3 up, mat4 dest);

CGLM_EXPORT
void
glmc_look(vec3 eye, vec3 dir, vec3 up, mat4 dest);

CGLM_EXPORT
void
glmc_look_anyup(vec3 eye, vec3 dir, mat4 dest);

CGLM_EXPORT
void
glmc_persp_decomp(mat4 proj,
                  float * __restrict nearVal,
                  float * __restrict farVal,
                  float * __restrict top,
                  float * __restrict bottom,
                  float * __restrict left,
                  float * __restrict right);

CGLM_EXPORT
void
glmc_persp_decompv(mat4 proj, float dest[6]);

CGLM_EXPORT
void
glmc_persp_decomp_x(mat4 proj,
                    float * __restrict left,
                    float * __restrict right);

CGLM_EXPORT
void
glmc_persp_decomp_y(mat4 proj,
                    float * __restrict top,
                    float * __restrict bottom);

CGLM_EXPORT
void
glmc_persp_decomp_z(mat4 proj,
                    float * __restrict nearVal,
                    float * __restrict farVal);

CGLM_EXPORT
void
glmc_persp_decomp_far(mat4 proj, float * __restrict farVal);

CGLM_EXPORT
void
glmc_persp_decomp_near(mat4 proj, float * __restrict nearVal);

CGLM_EXPORT
float
glmc_persp_fovy(mat4 proj);

CGLM_EXPORT
float
glmc_persp_aspect(mat4 proj);

CGLM_EXPORT
void
glmc_persp_sizes(mat4 proj, float fovy, vec4 dest);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_cam_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_quat_h
#define cglmc_quat_h
#ifdef __cplusplus
extern "C" {
#endif


CGLM_EXPORT
void
glmc_quat_identity(versor q);

CGLM_EXPORT
void
glmc_quat_identity_array(versor * __restrict q, size_t count);

CGLM_EXPORT
void
glmc_quat_init(versor q, float x, float y, float z, float w);

CGLM_EXPORT
void
glmc_quat(versor q, float angle, float x, float y, float z);

CGLM_EXPORT
void
glmc_quatv(versor q, float angle, vec3 axis);

CGLM_EXPORT
void
glmc_quat_copy(versor q, versor dest);

CGLM_EXPORT
float
glmc_quat_norm(versor q);

CGLM_EXPORT
void
glmc_quat_normalize_to(versor q, versor dest);

CGLM_EXPORT
void
glmc_quat_normalize(versor q);

CGLM_EXPORT
float
glmc_quat_dot(versor p, versor q);

CGLM_EXPORT
void
glmc_quat_conjugate(versor q, versor dest);

CGLM_EXPORT
void
glmc_quat_inv(versor q, versor dest);

CGLM_EXPORT
void
glmc_quat_add(versor p, versor q, versor dest);

CGLM_EXPORT
void
glmc_quat_sub(versor p, versor q, versor dest);

CGLM_EXPORT
float
glmc_quat_real(versor q);

CGLM_EXPORT
void
glmc_quat_imag(versor q, vec3 dest);

CGLM_EXPORT
void
glmc_quat_imagn(versor q, vec3 dest);

CGLM_EXPORT
float
glmc_quat_imaglen(versor q);

CGLM_EXPORT
float
glmc_quat_angle(versor q);

CGLM_EXPORT
void
glmc_quat_axis(versor q, vec3 dest);

CGLM_EXPORT
void
glmc_quat_mul(versor p, versor q, versor dest);

CGLM_EXPORT
void
glmc_quat_mat4(versor q, mat4 dest);

CGLM_EXPORT
void
glmc_quat_mat4t(versor q, mat4 dest);

CGLM_EXPORT
void
glmc_quat_mat3(versor q, mat3 dest);

CGLM_EXPORT
void
glmc_quat_mat3t(versor q, mat3 dest);

CGLM_EXPORT
void
glmc_quat_lerp(versor from, versor to, float t, versor dest);
    
CGLM_EXPORT
void
glmc_quat_lerpc(versor from, versor to, float t, versor dest);

CGLM_EXPORT
void
glmc_quat_slerp(versor q, versor r, float t, versor dest);

CGLM_EXPORT
void
glmc_quat_look(vec3 eye, versor ori, mat4 dest);

CGLM_EXPORT
void
glmc_quat_for(vec3 dir, vec3 up, versor dest);

CGLM_EXPORT
void
glmc_quat_forp(vec3 from, vec3 to, vec3 up, versor dest);

CGLM_EXPORT
void
glmc_quat_rotatev(versor from, vec3 to, vec3 dest);

CGLM_EXPORT
void
glmc_quat_rotate(mat4 m, versor q, mat4 dest);

CGLM_EXPORT
void
glmc_quat_rotate_at(mat4 model, versor q, vec3 pivot);

CGLM_EXPORT
void
glmc_quat_rotate_atm(mat4 m, versor q, vec3 pivot);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_quat_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_euler_h
#define cglmc_euler_h
#ifdef __cplusplus
extern "C" {
#endif


CGLM_EXPORT
void
glmc_euler_angles(mat4 m, vec3 dest);

CGLM_EXPORT
void
glmc_euler(vec3 angles, mat4 dest);

CGLM_EXPORT
void
glmc_euler_xyz(vec3 angles,  mat4 dest);

CGLM_EXPORT
void
glmc_euler_zyx(vec3 angles,  mat4 dest);

CGLM_EXPORT
void
glmc_euler_zxy(vec3 angles, mat4 dest);

CGLM_EXPORT
void
glmc_euler_xzy(vec3 angles, mat4 dest);

CGLM_EXPORT
void
glmc_euler_yzx(vec3 angles, mat4 dest);

CGLM_EXPORT
void
glmc_euler_yxz(vec3 angles, mat4 dest);

CGLM_EXPORT
void
glmc_euler_by_order(vec3 angles, glm_euler_seq axis, mat4 dest);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_euler_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_plane_h
#define cglmc_plane_h
#ifdef __cplusplus
extern "C" {
#endif


CGLM_EXPORT
void
glmc_plane_normalize(vec4 plane);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_plane_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_frustum_h
#define cglmc_frustum_h
#ifdef __cplusplus
extern "C" {
#endif


CGLM_EXPORT
void
glmc_frustum_planes(mat4 m, vec4 dest[6]);

CGLM_EXPORT
void
glmc_frustum_corners(mat4 invMat, vec4 dest[8]);

CGLM_EXPORT
void
glmc_frustum_center(vec4 corners[8], vec4 dest);

CGLM_EXPORT
void
glmc_frustum_box(vec4 corners[8], mat4 m, vec3 box[2]);

CGLM_EXPORT
void
glmc_frustum_corners_at(vec4  corners[8],
                        float splitDist,
                        float farDist,
                        vec4  planeCorners[4]);
#ifdef __cplusplus
}
#endif
#endif /* cglmc_frustum_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_box_h
#define cglmc_box_h
#ifdef __cplusplus
extern "C" {
#endif


CGLM_EXPORT
void
glmc_aabb_transform(vec3 box[2], mat4 m, vec3 dest[2]);

CGLM_EXPORT
void
glmc_aabb_merge(vec3 box1[2], vec3 box2[2], vec3 dest[2]);

CGLM_EXPORT
void
glmc_aabb_crop(vec3 box[2], vec3 cropBox[2], vec3 dest[2]);

CGLM_EXPORT
void
glmc_aabb_crop_until(vec3 box[2],
                     vec3 cropBox[2],
                     vec3 clampBox[2],
                     vec3 dest[2]);

CGLM_EXPORT
bool
glmc_aabb_frustum(vec3 box[2], vec4 planes[6]);

CGLM_EXPORT
void
glmc_aabb_invalidate(vec3 box[2]);

CGLM_EXPORT
bool
glmc_aabb_isvalid(vec3 box[2]);

CGLM_EXPORT
float
glmc_aabb_size(vec3 box[2]);

CGLM_EXPORT
float
glmc_aabb_radius(vec3 box[2]);

CGLM_EXPORT
void
glmc_aabb_center(vec3 box[2], vec3 dest);

CGLM_EXPORT
bool
glmc_aabb_aabb(vec3 box[2], vec3 other[2]);

CGLM_EXPORT
bool
glmc_aabb_point(vec3 box[2], vec3 point);

CGLM_EXPORT
bool
glmc_aabb_contains(vec3 box[2], vec3 other[2]);

CGLM_EXPORT
bool
glmc_aabb_sphere(vec3 box[2], vec4 s);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_box_h */


/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_io_h
#define cglmc_io_h

#ifdef __cplusplus
extern "C" {
#endif


CGLM_EXPORT
void
glmc_mat4_print(mat4   matrix,
                FILE * __restrict ostream);

CGLM_EXPORT
void
glmc_mat3_print(mat3 matrix,
                FILE * __restrict ostream);

CGLM_EXPORT
void
glmc_vec4_print(vec4 vec,
                FILE * __restrict ostream);

CGLM_EXPORT
void
glmc_vec3_print(vec3 vec,
                FILE * __restrict ostream);

CGLM_EXPORT
void
glmc_versor_print(versor vec,
                  FILE * __restrict ostream);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_io_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_project_h
#define cglmc_project_h
#ifdef __cplusplus
extern "C" {
#endif


CGLM_EXPORT
void
glmc_unprojecti(vec3 pos, mat4 invMat, vec4 vp, vec3 dest);

CGLM_EXPORT
void
glmc_unproject(vec3 pos, mat4 m, vec4 vp, vec3 dest);

CGLM_EXPORT
void
glmc_project(vec3 pos, mat4 m, vec4 vp, vec3 dest);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_project_h */



/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_sphere_h
#define cglmc_sphere_h
#ifdef __cplusplus
extern "C" {
#endif


CGLM_EXPORT
float
glmc_sphere_radii(vec4 s);

CGLM_EXPORT
void
glmc_sphere_transform(vec4 s, mat4 m, vec4 dest);

CGLM_EXPORT
void
glmc_sphere_merge(vec4 s1, vec4 s2, vec4 dest);

CGLM_EXPORT
bool
glmc_sphere_sphere(vec4 s1, vec4 s2);

CGLM_EXPORT
bool
glmc_sphere_point(vec4 s, vec3 point);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_sphere_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_ease_h
#define cglmc_ease_h
#ifdef __cplusplus
extern "C" {
#endif


CGLM_EXPORT
float
glmc_ease_linear(float t);

CGLM_EXPORT
float
glmc_ease_sine_in(float t);

CGLM_EXPORT
float
glmc_ease_sine_out(float t);

CGLM_EXPORT
float
glmc_ease_sine_inout(float t);

CGLM_EXPORT
float
glmc_ease_quad_in(float t);

CGLM_EXPORT
float
glmc_ease_quad_out(float t);

CGLM_EXPORT
float
glmc_ease_quad_inout(float t);

CGLM_EXPORT
float
glmc_ease_cubic_in(float t);

CGLM_EXPORT
float
glmc_ease_cubic_out(float t);

CGLM_EXPORT
float
glmc_ease_cubic_inout(float t);

CGLM_EXPORT
float
glmc_ease_quart_in(float t);

CGLM_EXPORT
float
glmc_ease_quart_out(float t);

CGLM_EXPORT
float
glmc_ease_quart_inout(float t);

CGLM_EXPORT
float
glmc_ease_quint_in(float t);

CGLM_EXPORT
float
glmc_ease_quint_out(float t);

CGLM_EXPORT
float
glmc_ease_quint_inout(float t);

CGLM_EXPORT
float
glmc_ease_exp_in(float t);

CGLM_EXPORT
float
glmc_ease_exp_out(float t);

CGLM_EXPORT
float
glmc_ease_exp_inout(float t);

CGLM_EXPORT
float
glmc_ease_circ_in(float t);

CGLM_EXPORT
float
glmc_ease_circ_out(float t);

CGLM_EXPORT
float
glmc_ease_circ_inout(float t);

CGLM_EXPORT
float
glmc_ease_back_in(float t);

CGLM_EXPORT
float
glmc_ease_back_out(float t);

CGLM_EXPORT
float
glmc_ease_back_inout(float t);

CGLM_EXPORT
float
glmc_ease_elast_in(float t);

CGLM_EXPORT
float
glmc_ease_elast_out(float t);

CGLM_EXPORT
float
glmc_ease_elast_inout(float t);

CGLM_EXPORT
float
glmc_ease_bounce_out(float t);

CGLM_EXPORT
float
glmc_ease_bounce_in(float t);

CGLM_EXPORT
float
glmc_ease_bounce_inout(float t);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_ease_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_curve_h
#define cglmc_curve_h
#ifdef __cplusplus
extern "C" {
#endif


CGLM_EXPORT
float
glmc_smc(float s, mat4 m, vec4 c);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_curve_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_bezier_h
#define cglmc_bezier_h
#ifdef __cplusplus
extern "C" {
#endif


CGLM_EXPORT
float
glmc_bezier(float s, float p0, float c0, float c1, float p1);

CGLM_EXPORT
float
glmc_hermite(float s, float p0, float t0, float t1, float p1);

CGLM_EXPORT
float
glmc_decasteljau(float prm, float p0, float c0, float c1, float p1);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_bezier_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_ray_h
#define cglmc_ray_h
#ifdef __cplusplus
extern "C" {
#endif

CGLM_EXPORT
bool
glmc_ray_triangle(vec3   origin,
                  vec3   direction,
                  vec3   v0,
                  vec3   v1,
                  vec3   v2,
                  float *d);
    
#ifdef __cplusplus
}
#endif
#endif /* cglmc_ray_h */

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#ifndef cglmc_affine2d_h
#define cglmc_affine2d_h
#ifdef __cplusplus
extern "C" {
#endif


CGLM_EXPORT
void
glmc_translate2d_make(mat3 m, vec2 v);

CGLM_EXPORT
void
glmc_translate2d_to(mat3 m, vec2 v, mat3 dest);

CGLM_EXPORT
void
glmc_translate2d(mat3 m, vec2 v);

CGLM_EXPORT
void
glmc_translate2d_x(mat3 m, float to);

CGLM_EXPORT
void
glmc_translate2d_y(mat3 m, float to);

CGLM_EXPORT
void
glmc_scale2d_to(mat3 m, vec2 v, mat3 dest);

CGLM_EXPORT
void
glmc_scale2d_make(mat3 m, vec2 v);

CGLM_EXPORT
void
glmc_scale2d(mat3 m, vec2 v);

CGLM_EXPORT
void
glmc_scale2d_uni(mat3 m, float s);

CGLM_EXPORT
void
glmc_rotate2d_make(mat3 m, float angle);

CGLM_EXPORT
void
glmc_rotate2d(mat3 m, float angle);

CGLM_EXPORT
void
glmc_rotate2d_to(mat3 m, float angle, mat3 dest);

#ifdef __cplusplus
}
#endif
#endif /* cglmc_affine2d_h */


#ifdef __cplusplus
}
#endif
#endif /* cglm_call_h */


CGLM_EXPORT
void
glmc_quat_identity(versor q) {
  glm_quat_identity(q);
}

CGLM_EXPORT
void
glmc_quat_identity_array(versor * __restrict q, size_t count) {
  glm_quat_identity_array(q, count);
}

CGLM_EXPORT
void
glmc_quat_init(versor q, float x, float y, float z, float w) {
  glm_quat_init(q, x, y, z, w);
}

CGLM_EXPORT
void
glmc_quat(versor q, float angle, float x, float y, float z) {
  glm_quat(q, angle, x, y, z);
}

CGLM_EXPORT
void
glmc_quatv(versor q, float angle, vec3 axis) {
  glm_quatv(q, angle, axis);
}

CGLM_EXPORT
void
glmc_quat_copy(versor q, versor dest) {
  glm_quat_copy(q, dest);
}

CGLM_EXPORT
float
glmc_quat_norm(versor q) {
  return glm_quat_norm(q);
}

CGLM_EXPORT
void
glmc_quat_normalize_to(versor q, versor dest) {
  glm_quat_normalize_to(q, dest);
}

CGLM_EXPORT
void
glmc_quat_normalize(versor q) {
  glm_quat_normalize(q);
}

CGLM_EXPORT
float
glmc_quat_dot(versor p, versor q) {
  return glm_quat_dot(p, q);
}

CGLM_EXPORT
void
glmc_quat_conjugate(versor q, versor dest) {
  glm_quat_conjugate(q, dest);
}

CGLM_EXPORT
void
glmc_quat_inv(versor q, versor dest) {
  glm_quat_inv(q, dest);
}

CGLM_EXPORT
void
glmc_quat_add(versor p, versor q, versor dest) {
  glm_quat_add(p, q, dest);
}

CGLM_EXPORT
void
glmc_quat_sub(versor p, versor q, versor dest) {
  glm_quat_sub(p, q, dest);
}

CGLM_EXPORT
float
glmc_quat_real(versor q) {
  return glm_quat_real(q);
}

CGLM_EXPORT
void
glmc_quat_imag(versor q, vec3 dest) {
  glm_quat_imag(q, dest);
}

CGLM_EXPORT
void
glmc_quat_imagn(versor q, vec3 dest) {
  glm_quat_imagn(q, dest);
}

CGLM_EXPORT
float
glmc_quat_imaglen(versor q) {
  return glm_quat_imaglen(q);
}

CGLM_EXPORT
float
glmc_quat_angle(versor q) {
  return glm_quat_angle(q);
}

CGLM_EXPORT
void
glmc_quat_axis(versor q, vec3 dest) {
  glm_quat_axis(q, dest);
}

CGLM_EXPORT
void
glmc_quat_mul(versor p, versor q, versor dest) {
  glm_quat_mul(p, q, dest);
}

CGLM_EXPORT
void
glmc_quat_mat4(versor q, mat4 dest) {
  glm_quat_mat4(q, dest);
}

CGLM_EXPORT
void
glmc_quat_mat4t(versor q, mat4 dest) {
  glm_quat_mat4t(q, dest);
}

CGLM_EXPORT
void
glmc_quat_mat3(versor q, mat3 dest) {
  glm_quat_mat3(q, dest);
}

CGLM_EXPORT
void
glmc_quat_mat3t(versor q, mat3 dest) {
  glm_quat_mat3t(q, dest);
}

CGLM_EXPORT
void
glmc_quat_lerp(versor from, versor to, float t, versor dest) {
  glm_quat_lerp(from, to, t, dest);
}

CGLM_EXPORT
void
glmc_quat_lerpc(versor from, versor to, float t, versor dest) {
  glm_quat_lerpc(from, to, t, dest);
}

CGLM_EXPORT
void
glmc_quat_slerp(versor from, versor to, float t, versor dest) {
  glm_quat_slerp(from, to, t, dest);
}

CGLM_EXPORT
void
glmc_quat_look(vec3 eye, versor ori, mat4 dest) {
  glm_quat_look(eye, ori, dest);
}

CGLM_EXPORT
void
glmc_quat_for(vec3 dir, vec3 up, versor dest) {
  glm_quat_for(dir, up, dest);
}

CGLM_EXPORT
void
glmc_quat_forp(vec3 from, vec3 to, vec3 up, versor dest) {
  glm_quat_forp(from, to, up, dest);
}

CGLM_EXPORT
void
glmc_quat_rotatev(versor q, vec3 v, vec3 dest) {
  glm_quat_rotatev(q, v, dest);
}

CGLM_EXPORT
void
glmc_quat_rotate(mat4 m, versor q, mat4 dest) {
  glm_quat_rotate(m, q, dest);
}

CGLM_EXPORT
void
glmc_quat_rotate_at(mat4 model, versor q, vec3 pivot) {
  glm_quat_rotate_at(model, q, pivot);
}

CGLM_EXPORT
void
glmc_quat_rotate_atm(mat4 m, versor q, vec3 pivot) {
  glm_quat_rotate_atm(m, q, pivot);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
float
glmc_ease_linear(float t) {
  return glm_ease_linear(t);
}

CGLM_EXPORT
float
glmc_ease_sine_in(float t) {
  return glm_ease_sine_in(t);
}

CGLM_EXPORT
float
glmc_ease_sine_out(float t) {
  return glm_ease_sine_out(t);
}

CGLM_EXPORT
float
glmc_ease_sine_inout(float t) {
  return glm_ease_sine_inout(t);
}

CGLM_EXPORT
float
glmc_ease_quad_in(float t) {
  return glm_ease_quad_in(t);
}

CGLM_EXPORT
float
glmc_ease_quad_out(float t) {
  return glm_ease_quad_out(t);
}

CGLM_EXPORT
float
glmc_ease_quad_inout(float t) {
  return glm_ease_quad_inout(t);
}

CGLM_EXPORT
float
glmc_ease_cubic_in(float t) {
  return glm_ease_cubic_in(t);
}

CGLM_EXPORT
float
glmc_ease_cubic_out(float t) {
  return glm_ease_cubic_out(t);
}

CGLM_EXPORT
float
glmc_ease_cubic_inout(float t) {
  return glm_ease_cubic_inout(t);
}

CGLM_EXPORT
float
glmc_ease_quart_in(float t) {
  return glm_ease_quart_in(t);
}

CGLM_EXPORT
float
glmc_ease_quart_out(float t) {
  return glm_ease_quart_out(t);
}

CGLM_EXPORT
float
glmc_ease_quart_inout(float t) {
  return glm_ease_quart_inout(t);
}

CGLM_EXPORT
float
glmc_ease_quint_in(float t) {
  return glm_ease_quint_in(t);
}

CGLM_EXPORT
float
glmc_ease_quint_out(float t) {
  return glm_ease_quint_out(t);
}

CGLM_EXPORT
float
glmc_ease_quint_inout(float t) {
  return glm_ease_quint_inout(t);
}

CGLM_EXPORT
float
glmc_ease_exp_in(float t) {
  return glm_ease_exp_in(t);
}

CGLM_EXPORT
float
glmc_ease_exp_out(float t) {
  return glm_ease_exp_out(t);
}

CGLM_EXPORT
float
glmc_ease_exp_inout(float t) {
  return glm_ease_exp_inout(t);
}

CGLM_EXPORT
float
glmc_ease_circ_in(float t) {
  return glm_ease_circ_in(t);
}

CGLM_EXPORT
float
glmc_ease_circ_out(float t) {
  return glm_ease_circ_out(t);
}

CGLM_EXPORT
float
glmc_ease_circ_inout(float t) {
  return glm_ease_circ_inout(t);
}

CGLM_EXPORT
float
glmc_ease_back_in(float t) {
  return glm_ease_back_in(t);
}

CGLM_EXPORT
float
glmc_ease_back_out(float t) {
  return glm_ease_back_out(t);
}

CGLM_EXPORT
float
glmc_ease_back_inout(float t) {
  return glm_ease_back_inout(t);
}

CGLM_EXPORT
float
glmc_ease_elast_in(float t) {
  return glm_ease_elast_in(t);
}

CGLM_EXPORT
float
glmc_ease_elast_out(float t) {
  return glm_ease_elast_out(t);
}

CGLM_EXPORT
float
glmc_ease_elast_inout(float t) {
  return glm_ease_elast_inout(t);
}

CGLM_EXPORT
float
glmc_ease_bounce_out(float t) {
  return glm_ease_bounce_out(t);
}

CGLM_EXPORT
float
glmc_ease_bounce_in(float t) {
  return glm_ease_bounce_in(t);
}

CGLM_EXPORT
float
glmc_ease_bounce_inout(float t) {
  return glm_ease_bounce_inout(t);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
void
glmc_vec4(vec3 v3, float last, vec4 dest) {
  glm_vec4(v3, last, dest);
}

CGLM_EXPORT
void
glmc_vec4_zero(vec4 v) {
  glm_vec4_zero(v);
}

CGLM_EXPORT
void
glmc_vec4_one(vec4 v) {
  glm_vec4_one(v);
}

CGLM_EXPORT
void
glmc_vec4_copy3(vec4 v, vec3 dest) {
  glm_vec4_copy3(v, dest);
}

CGLM_EXPORT
void
glmc_vec4_copy(vec4 v, vec4 dest) {
  glm_vec4_copy(v, dest);
}

CGLM_EXPORT
void
glmc_vec4_ucopy(vec4 v, vec4 dest) {
  glm_vec4_ucopy(v, dest);
}

CGLM_EXPORT
float
glmc_vec4_dot(vec4 a, vec4 b) {
  return glm_vec4_dot(a, b);
}

CGLM_EXPORT
float
glmc_vec4_norm(vec4 v) {
  return glm_vec4_norm(v);
}

CGLM_EXPORT
void
glmc_vec4_normalize_to(vec4 v, vec4 dest) {
  glm_vec4_normalize_to(v, dest);
}

CGLM_EXPORT
void
glmc_vec4_normalize(vec4 v) {
  glm_vec4_normalize(v);
}

CGLM_EXPORT
float
glmc_vec4_norm2(vec4 v) {
  return glm_vec4_norm2(v);
}

CGLM_EXPORT
float
glmc_vec4_norm_one(vec4 v) {
  return glm_vec4_norm_one(v);
}

CGLM_EXPORT
float
glmc_vec4_norm_inf(vec4 v) {
  return glm_vec4_norm_inf(v);
}

CGLM_EXPORT
void
glmc_vec4_add(vec4 a, vec4 b, vec4 dest) {
  glm_vec4_add(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec4_adds(vec4 v, float s, vec4 dest) {
  glm_vec4_adds(v, s, dest);
}

CGLM_EXPORT
void
glmc_vec4_sub(vec4 a, vec4 b, vec4 dest) {
  glm_vec4_sub(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec4_subs(vec4 v, float s, vec4 dest) {
  glm_vec4_subs(v, s, dest);
}

CGLM_EXPORT
void
glmc_vec4_mul(vec4 a, vec4 b, vec4 d) {
  glm_vec4_mul(a, b, d);
}

CGLM_EXPORT
void
glmc_vec4_scale(vec4 v, float s, vec4 dest) {
  glm_vec4_scale(v, s, dest);
}

CGLM_EXPORT
void
glmc_vec4_scale_as(vec4 v, float s, vec4 dest) {
  glm_vec4_scale_as(v, s, dest);
}

CGLM_EXPORT
void
glmc_vec4_div(vec4 a, vec4 b, vec4 dest) {
  glm_vec4_div(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec4_divs(vec4 v, float s, vec4 dest) {
  glm_vec4_divs(v, s, dest);
}

CGLM_EXPORT
void
glmc_vec4_addadd(vec4 a, vec4 b, vec4 dest) {
  glm_vec4_addadd(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec4_subadd(vec4 a, vec4 b, vec4 dest) {
  glm_vec4_subadd(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec4_muladd(vec4 a, vec4 b, vec4 dest) {
  glm_vec4_muladd(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec4_muladds(vec4 a, float s, vec4 dest) {
  glm_vec4_muladds(a, s, dest);
}

CGLM_EXPORT
void
glmc_vec4_maxadd(vec4 a, vec4 b, vec4 dest) {
  glm_vec4_maxadd(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec4_minadd(vec4 a, vec4 b, vec4 dest) {
  glm_vec4_minadd(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec4_negate(vec4 v) {
  glm_vec4_negate(v);
}

CGLM_EXPORT
void
glmc_vec4_negate_to(vec4 v, vec4 dest) {
  glm_vec4_negate_to(v, dest);
}

CGLM_EXPORT
float
glmc_vec4_distance(vec4 a, vec4 b) {
  return glm_vec4_distance(a, b);
}

CGLM_EXPORT
float
glmc_vec4_distance2(vec4 a, vec4 b) {
  return glm_vec4_distance2(a, b);
}

CGLM_EXPORT
void
glmc_vec4_maxv(vec4 a, vec4 b, vec4 dest) {
  glm_vec4_maxv(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec4_minv(vec4 a, vec4 b, vec4 dest) {
  glm_vec4_minv(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec4_clamp(vec4 v, float minVal, float maxVal) {
  glm_vec4_clamp(v, minVal, maxVal);
}

CGLM_EXPORT
void
glmc_vec4_lerp(vec4 from, vec4 to, float t, vec4 dest) {
  glm_vec4_lerp(from, to, t, dest);
}

CGLM_EXPORT
void
glmc_vec4_lerpc(vec4 from, vec4 to, float t, vec4 dest) {
  glm_vec4_lerpc(from, to, t, dest);
}

CGLM_EXPORT
void
glmc_vec4_step_uni(float edge, vec4 x, vec4 dest) {
  glm_vec4_step_uni(edge, x, dest);
}

CGLM_EXPORT
void
glmc_vec4_step(vec4 edge, vec4 x, vec4 dest) {
  glm_vec4_step(edge, x, dest);
}

CGLM_EXPORT
void
glmc_vec4_smoothstep_uni(float edge0, float edge1, vec4 x, vec4 dest) {
  glm_vec4_smoothstep_uni(edge0, edge1, x, dest);
}

CGLM_EXPORT
void
glmc_vec4_smoothstep(vec4 edge0, vec4 edge1, vec4 x, vec4 dest) {
  glm_vec4_smoothstep(edge0, edge1, x, dest);
}

CGLM_EXPORT
void
glmc_vec4_smoothinterp(vec4 from, vec4 to, float t, vec4 dest) {
  glm_vec4_smoothinterp(from, to, t, dest);
}

CGLM_EXPORT
void
glmc_vec4_smoothinterpc(vec4 from, vec4 to, float t, vec4 dest) {
  glm_vec4_smoothinterpc(from, to, t, dest);
}

CGLM_EXPORT
void
glmc_vec4_cubic(float s, vec4 dest) {
  glm_vec4_cubic(s, dest);
}

/* ext */

CGLM_EXPORT
void
glmc_vec4_mulv(vec4 a, vec4 b, vec4 d) {
  glm_vec4_mulv(a, b, d);
}

CGLM_EXPORT
void
glmc_vec4_broadcast(float val, vec4 d) {
  glm_vec4_broadcast(val, d);
}

CGLM_EXPORT
void
glmc_vec4_fill(vec4 v, float val) {
  glm_vec4_fill(v, val);
}

CGLM_EXPORT
bool
glmc_vec4_eq(vec4 v, float val) {
  return glm_vec4_eq(v, val);
}

CGLM_EXPORT
bool
glmc_vec4_eq_eps(vec4 v, float val) {
  return glm_vec4_eq_eps(v, val);
}

CGLM_EXPORT
bool
glmc_vec4_eq_all(vec4 v) {
  return glm_vec4_eq_all(v);
}

CGLM_EXPORT
bool
glmc_vec4_eqv(vec4 a, vec4 b) {
  return glm_vec4_eqv(a, b);
}

CGLM_EXPORT
bool
glmc_vec4_eqv_eps(vec4 a, vec4 b) {
  return glm_vec4_eqv_eps(a, b);
}

CGLM_EXPORT
float
glmc_vec4_max(vec4 v) {
  return glm_vec4_max(v);
}

CGLM_EXPORT
float
glmc_vec4_min(vec4 v) {
  return glm_vec4_min(v);
}

CGLM_EXPORT
bool
glmc_vec4_isnan(vec4 v) {
  return glm_vec4_isnan(v);
}

CGLM_EXPORT
bool
glmc_vec4_isinf(vec4 v) {
  return glm_vec4_isinf(v);
}

CGLM_EXPORT
bool
glmc_vec4_isvalid(vec4 v) {
  return glm_vec4_isvalid(v);
}

CGLM_EXPORT
void
glmc_vec4_sign(vec4 v, vec4 dest) {
  glm_vec4_sign(v, dest);
}

CGLM_EXPORT
void
glmc_vec4_abs(vec4 v, vec4 dest) {
  glm_vec4_abs(v, dest);
}

CGLM_EXPORT
void
glmc_vec4_fract(vec4 v, vec4 dest) {
  glm_vec4_fract(v, dest);
}

CGLM_EXPORT
float
glmc_vec4_hadd(vec4 v) {
  return glm_vec4_hadd(v);
}

CGLM_EXPORT
void
glmc_vec4_sqrt(vec4 v, vec4 dest) {
  glm_vec4_sqrt(v, dest);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
void
glmc_translate_make(mat4 m, vec3 v) {
  glm_translate_make(m, v);
}

CGLM_EXPORT
void
glmc_translate_to(mat4 m, vec3 v, mat4 dest) {
  glm_translate_to(m, v, dest);
}

CGLM_EXPORT
void
glmc_translate(mat4 m, vec3 v) {
  glm_translate(m, v);
}

CGLM_EXPORT
void
glmc_translate_x(mat4 m, float to) {
  glm_translate_x(m, to);
}

CGLM_EXPORT
void
glmc_translate_y(mat4 m, float to) {
  glm_translate_y(m, to);
}

CGLM_EXPORT
void
glmc_translate_z(mat4 m, float to) {
  glm_translate_z(m, to);
}

CGLM_EXPORT
void
glmc_scale_make(mat4 m, vec3 v) {
  glm_scale_make(m, v);
}

CGLM_EXPORT
void
glmc_scale_to(mat4 m, vec3 v, mat4 dest) {
  glm_scale_to(m, v, dest);
}

CGLM_EXPORT
void
glmc_scale(mat4 m, vec3 v) {
  glm_scale(m, v);
}

CGLM_EXPORT
void
glmc_scale_uni(mat4 m, float s) {
  glm_scale_uni(m, s);
}

CGLM_EXPORT
void
glmc_rotate_x(mat4 m, float rad, mat4 dest) {
  glm_rotate_x(m, rad, dest);
}

CGLM_EXPORT
void
glmc_rotate_y(mat4 m, float rad, mat4 dest) {
  glm_rotate_y(m, rad, dest);
}

CGLM_EXPORT
void
glmc_rotate_z(mat4 m, float rad, mat4 dest) {
  glm_rotate_z(m, rad, dest);
}

CGLM_EXPORT
void
glmc_rotate_make(mat4 m, float angle, vec3 axis) {
  glm_rotate_make(m, angle, axis);
}

CGLM_EXPORT
void
glmc_rotate(mat4 m, float angle, vec3 axis) {
  glm_rotate(m, angle, axis);
}

CGLM_EXPORT
void
glmc_rotate_at(mat4 m, vec3 pivot, float angle, vec3 axis) {
  glm_rotate_at(m, pivot, angle, axis);
}

CGLM_EXPORT
void
glmc_rotate_atm(mat4 m, vec3 pivot, float angle, vec3 axis) {
  glm_rotate_atm(m, pivot, angle, axis);
}

CGLM_EXPORT
void
glmc_decompose_scalev(mat4 m, vec3 s) {
  glm_decompose_scalev(m, s);
}

CGLM_EXPORT
bool
glmc_uniscaled(mat4 m) {
  return glm_uniscaled(m);
}

CGLM_EXPORT
void
glmc_decompose_rs(mat4 m, mat4 r, vec3 s) {
  glm_decompose_rs(m, r, s);
}

CGLM_EXPORT
void
glmc_decompose(mat4 m, vec4 t, mat4 r, vec3 s) {
  glm_decompose(m, t, r, s);
}

CGLM_EXPORT
void
glmc_mul(mat4 m1, mat4 m2, mat4 dest) {
  glm_mul(m1, m2, dest);
}

CGLM_EXPORT
void
glmc_mul_rot(mat4 m1, mat4 m2, mat4 dest) {
  glm_mul_rot(m1, m2, dest);
}

CGLM_EXPORT
void
glmc_inv_tr(mat4 mat) {
  glm_inv_tr(mat);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
void
glmc_mat2_copy(mat2 mat, mat2 dest) {
  glm_mat2_copy(mat, dest);
}

CGLM_EXPORT
void
glmc_mat2_identity(mat2 mat) {
  glm_mat2_identity(mat);
}

CGLM_EXPORT
void
glmc_mat2_identity_array(mat2 * __restrict mat, size_t count) {
  glm_mat2_identity_array(mat, count);
}

CGLM_EXPORT
void
glmc_mat2_zero(mat2 mat) {
  glm_mat2_zero(mat);
}

CGLM_EXPORT
void
glmc_mat2_mul(mat2 m1, mat2 m2, mat2 dest) {
  glm_mat2_mul(m1, m2, dest);
}

CGLM_EXPORT
void
glmc_mat2_transpose_to(mat2 m, mat2 dest) {
  glm_mat2_transpose_to(m, dest);
}

CGLM_EXPORT
void
glmc_mat2_transpose(mat2 m) {
  glm_mat2_transpose(m);
}

CGLM_EXPORT
void
glmc_mat2_mulv(mat2 m, vec2 v, vec2 dest) {
  glm_mat2_mulv(m, v, dest);
}

CGLM_EXPORT
float
glmc_mat2_trace(mat2 m) {
  return glm_mat2_trace(m);
}

CGLM_EXPORT
void
glmc_mat2_scale(mat2 m, float s) {
  glm_mat2_scale(m, s);
}

CGLM_EXPORT
float
glmc_mat2_det(mat2 mat) {
  return glm_mat2_det(mat);
}

CGLM_EXPORT
void
glmc_mat2_inv(mat2 mat, mat2 dest) {
  glm_mat2_inv(mat, dest);
}

CGLM_EXPORT
void
glmc_mat2_swap_col(mat2 mat, int col1, int col2) {
  glm_mat2_swap_col(mat, col1, col2);
}

CGLM_EXPORT
void
glmc_mat2_swap_row(mat2 mat, int row1, int row2) {
  glm_mat2_swap_row(mat, row1, row2);
}

CGLM_EXPORT
float
glmc_mat2_rmc(vec2 r, mat2 m, vec2 c) {
  return glm_mat2_rmc(r, m, c);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
float
glmc_smc(float s, mat4 m, vec4 c) {
  return glm_smc(s, m, c);
}


CGLM_EXPORT
bool
glmc_ray_triangle(vec3   origin,
                  vec3   direction,
                  vec3   v0,
                  vec3   v1,
                  vec3   v2,
                  float *d) {
    return glm_ray_triangle(origin, direction, v0, v1, v2, d);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */

#define CGLM_LIB_SRC


CGLM_EXPORT
void
glmc_mat4_print(mat4   matrix,
                FILE * __restrict ostream) {
  glm_mat4_print(matrix, ostream);
}

CGLM_EXPORT
void
glmc_mat3_print(mat3 matrix,
                FILE * __restrict ostream) {
  glm_mat3_print(matrix, ostream);
}

CGLM_EXPORT
void
glmc_vec4_print(vec4 vec,
                FILE * __restrict ostream) {
  glm_vec4_print(vec, ostream);
}

CGLM_EXPORT
void
glmc_vec3_print(vec3 vec,
                FILE * __restrict ostream) {
  glm_vec3_print(vec, ostream);
}

CGLM_EXPORT
void
glmc_versor_print(versor vec,
                  FILE * __restrict ostream) {
  glm_versor_print(vec, ostream);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
void
glmc_translate2d_make(mat3 m, vec2 v) {
  glm_translate2d_make(m, v);
}

CGLM_EXPORT
void
glmc_translate2d_to(mat3 m, vec2 v, mat3 dest) {
  glm_translate2d_to(m, v, dest);
}

CGLM_EXPORT
void
glmc_translate2d(mat3 m, vec2 v) {
  glm_translate2d(m, v);
}

CGLM_EXPORT
void
glmc_translate2d_x(mat3 m, float to) {
  glm_translate2d_x(m, to);
}

CGLM_EXPORT
void
glmc_translate2d_y(mat3 m, float to) {
  glm_translate2d_y(m, to);
}

CGLM_EXPORT
void
glmc_scale2d_to(mat3 m, vec2 v, mat3 dest) {
  glm_scale2d_to(m, v, dest);
}

CGLM_EXPORT
void
glmc_scale2d_make(mat3 m, vec2 v) {
  glm_scale2d_make(m, v);
}

CGLM_EXPORT
void
glmc_scale2d(mat3 m, vec2 v) {
  glm_scale2d(m, v);
}

CGLM_EXPORT
void
glmc_scale2d_uni(mat3 m, float s) {
  glm_scale2d_uni(m, s);
}

CGLM_EXPORT
void
glmc_rotate2d_make(mat3 m, float angle) {
  glm_rotate2d_make(m, angle);
}

CGLM_EXPORT
void
glmc_rotate2d(mat3 m, float angle) {
  glm_rotate2d(m, angle);
}

CGLM_EXPORT
void
glmc_rotate2d_to(mat3 m, float angle, mat3 dest) {
  glm_rotate2d_to(m, angle, dest);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
void
glmc_vec3(vec4 v4, vec3 dest) {
  glm_vec3(v4, dest);
}

CGLM_EXPORT
void
glmc_vec3_copy(vec3 a, vec3 dest) {
  glm_vec3_copy(a, dest);
}

CGLM_EXPORT
void
glmc_vec3_zero(vec3 v) {
  glm_vec3_zero(v);
}

CGLM_EXPORT
void
glmc_vec3_one(vec3 v) {
  glm_vec3_one(v);
}

CGLM_EXPORT
float
glmc_vec3_dot(vec3 a, vec3 b) {
  return glm_vec3_dot(a, b);
}

CGLM_EXPORT
void
glmc_vec3_cross(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_cross(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec3_crossn(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_crossn(a, b, dest);
}

CGLM_EXPORT
float
glmc_vec3_norm(vec3 v) {
  return glm_vec3_norm(v);
}

CGLM_EXPORT
void
glmc_vec3_normalize_to(vec3 v, vec3 dest) {
  glm_vec3_normalize_to(v, dest);
}

CGLM_EXPORT
void
glmc_vec3_normalize(vec3 v) {
  glm_vec3_normalize(v);
}

CGLM_EXPORT
float
glmc_vec3_norm2(vec3 v) {
  return glm_vec3_norm2(v);
}

CGLM_EXPORT
float
glmc_vec3_norm_one(vec3 v) {
  return glm_vec3_norm_one(v);
}

CGLM_EXPORT
float
glmc_vec3_norm_inf(vec3 v) {
  return glm_vec3_norm_inf(v);
}

CGLM_EXPORT
void
glmc_vec3_add(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_add(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec3_adds(vec3 v, float s, vec3 dest) {
  glm_vec3_adds(v, s, dest);
}

CGLM_EXPORT
void
glmc_vec3_sub(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_sub(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec3_subs(vec3 v, float s, vec3 dest) {
  glm_vec3_subs(v, s, dest);
}

CGLM_EXPORT
void
glmc_vec3_mul(vec3 a, vec3 b, vec3 d) {
  glm_vec3_mul(a, b, d);
}

CGLM_EXPORT
void
glmc_vec3_scale(vec3 v, float s, vec3 dest) {
  glm_vec3_scale(v, s, dest);
}

CGLM_EXPORT
void
glmc_vec3_scale_as(vec3 v, float s, vec3 dest) {
  glm_vec3_scale_as(v, s, dest);
}

CGLM_EXPORT
void
glmc_vec3_div(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_div(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec3_divs(vec3 a, float s, vec3 dest) {
  glm_vec3_divs(a, s, dest);
}

CGLM_EXPORT
void
glmc_vec3_addadd(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_addadd(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec3_subadd(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_subadd(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec3_muladd(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_muladd(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec3_muladds(vec3 a, float s, vec3 dest) {
  glm_vec3_muladds(a, s, dest);
}

CGLM_EXPORT
void
glmc_vec3_maxadd(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_maxadd(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec3_minadd(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_minadd(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec3_negate(vec3 v) {
  glm_vec3_negate(v);
}

CGLM_EXPORT
void
glmc_vec3_negate_to(vec3 v, vec3 dest) {
  glm_vec3_negate_to(v, dest);
}

CGLM_EXPORT
float
glmc_vec3_angle(vec3 a, vec3 b) {
  return glm_vec3_angle(a, b);
}

CGLM_EXPORT
void
glmc_vec3_rotate(vec3 v, float angle, vec3 axis) {
  glm_vec3_rotate(v, angle, axis);
}

CGLM_EXPORT
void
glmc_vec3_rotate_m4(mat4 m, vec3 v, vec3 dest) {
  glm_vec3_rotate_m4(m, v, dest);
}

CGLM_EXPORT
void
glmc_vec3_rotate_m3(mat3 m, vec3 v, vec3 dest) {
  glm_vec3_rotate_m3(m, v, dest);
}

CGLM_EXPORT
void
glmc_vec3_proj(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_proj(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec3_center(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_center(a, b, dest);
}

CGLM_EXPORT
float
glmc_vec3_distance(vec3 a, vec3 b) {
  return glm_vec3_distance(a, b);
}

CGLM_EXPORT
float
glmc_vec3_distance2(vec3 a, vec3 b) {
  return glm_vec3_distance2(a, b);
}

CGLM_EXPORT
void
glmc_vec3_maxv(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_maxv(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec3_minv(vec3 a, vec3 b, vec3 dest) {
  glm_vec3_minv(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec3_clamp(vec3 v, float minVal, float maxVal) {
  glm_vec3_clamp(v, minVal, maxVal);
}

CGLM_EXPORT
void
glmc_vec3_ortho(vec3 v, vec3 dest) {
  glm_vec3_ortho(v, dest);
}

CGLM_EXPORT
void
glmc_vec3_lerp(vec3 from, vec3 to, float t, vec3 dest) {
  glm_vec3_lerp(from, to, t, dest);
}

CGLM_EXPORT
void
glmc_vec3_lerpc(vec3 from, vec3 to, float t, vec3 dest) {
  glm_vec3_lerpc(from, to, t, dest);
}

CGLM_EXPORT
void
glmc_vec3_step_uni(float edge, vec3 x, vec3 dest) {
  glm_vec3_step_uni(edge, x, dest);
}

CGLM_EXPORT
void
glmc_vec3_step(vec3 edge, vec3 x, vec3 dest) {
  glm_vec3_step(edge, x, dest);
}

CGLM_EXPORT
void
glmc_vec3_smoothstep_uni(float edge0, float edge1, vec3 x, vec3 dest) {
  glm_vec3_smoothstep_uni(edge0, edge1, x, dest);
}

CGLM_EXPORT
void
glmc_vec3_smoothstep(vec3 edge0, vec3 edge1, vec3 x, vec3 dest) {
  glm_vec3_smoothstep(edge0, edge1, x, dest);
}

CGLM_EXPORT
void
glmc_vec3_smoothinterp(vec3 from, vec3 to, float t, vec3 dest) {
  glm_vec3_smoothinterp(from, to, t, dest);
}

CGLM_EXPORT
void
glmc_vec3_smoothinterpc(vec3 from, vec3 to, float t, vec3 dest) {
  glm_vec3_smoothinterpc(from, to, t, dest);
}

/* ext */

CGLM_EXPORT
void
glmc_vec3_mulv(vec3 a, vec3 b, vec3 d) {
  glm_vec3_mulv(a, b, d);
}

CGLM_EXPORT
void
glmc_vec3_broadcast(float val, vec3 d) {
  glm_vec3_broadcast(val, d);
}

CGLM_EXPORT
void
glmc_vec3_fill(vec3 v, float val) {
  glm_vec3_fill(v, val);
}

CGLM_EXPORT
bool
glmc_vec3_eq(vec3 v, float val) {
  return glm_vec3_eq(v, val);
}

CGLM_EXPORT
bool
glmc_vec3_eq_eps(vec3 v, float val) {
  return glm_vec3_eq_eps(v, val);
}

CGLM_EXPORT
bool
glmc_vec3_eq_all(vec3 v) {
  return glm_vec3_eq_all(v);
}

CGLM_EXPORT
bool
glmc_vec3_eqv(vec3 a, vec3 b) {
  return glm_vec3_eqv(a, b);
}

CGLM_EXPORT
bool
glmc_vec3_eqv_eps(vec3 a, vec3 b) {
  return glm_vec3_eqv_eps(a, b);
}

CGLM_EXPORT
float
glmc_vec3_max(vec3 v) {
  return glm_vec3_max(v);
}

CGLM_EXPORT
float
glmc_vec3_min(vec3 v) {
  return glm_vec3_min(v);
}

CGLM_EXPORT
bool
glmc_vec3_isnan(vec3 v) {
  return glm_vec3_isnan(v);
}

CGLM_EXPORT
bool
glmc_vec3_isinf(vec3 v) {
  return glm_vec3_isinf(v);
}

CGLM_EXPORT
bool
glmc_vec3_isvalid(vec3 v) {
  return glm_vec3_isvalid(v);
}

CGLM_EXPORT
void
glmc_vec3_sign(vec3 v, vec3 dest) {
  glm_vec3_sign(v, dest);
}

CGLM_EXPORT
void
glmc_vec3_abs(vec3 v, vec3 dest) {
  glm_vec3_abs(v, dest);
}

CGLM_EXPORT
void
glmc_vec3_fract(vec3 v, vec3 dest) {
  glm_vec3_fract(v, dest);
}

CGLM_EXPORT
float
glmc_vec3_hadd(vec3 v) {
  return glm_vec3_hadd(v);
}

CGLM_EXPORT
void
glmc_vec3_sqrt(vec3 v, vec3 dest) {
  glm_vec3_sqrt(v, dest);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
void
glmc_unprojecti(vec3 pos, mat4 invMat, vec4 vp, vec3 dest) {
  glm_unprojecti(pos, invMat, vp, dest);
}

CGLM_EXPORT
void
glmc_unproject(vec3 pos, mat4 m, vec4 vp, vec3 dest) {
  glm_unproject(pos, m, vp, dest);
}

CGLM_EXPORT
void
glmc_project(vec3 pos, mat4 m, vec4 vp, vec3 dest) {
  glm_project(pos, m, vp, dest);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
void
glmc_euler_angles(mat4 m, vec3 dest) {
  glm_euler_angles(m, dest);
}

CGLM_EXPORT
void
glmc_euler(vec3 angles, mat4 dest) {
  glm_euler(angles, dest);
}

CGLM_EXPORT
void
glmc_euler_xyz(vec3 angles,  mat4 dest) {
  glm_euler_xyz(angles, dest);
}

CGLM_EXPORT
void
glmc_euler_zyx(vec3 angles,  mat4 dest) {
  glm_euler_zyx(angles, dest);
}

CGLM_EXPORT
void
glmc_euler_zxy(vec3 angles, mat4 dest) {
  glm_euler_zxy(angles, dest);
}

CGLM_EXPORT
void
glmc_euler_xzy(vec3 angles, mat4 dest) {
  glm_euler_xzy(angles, dest);
}

CGLM_EXPORT
void
glmc_euler_yzx(vec3 angles, mat4 dest) {
  glm_euler_yzx(angles, dest);
}

CGLM_EXPORT
void
glmc_euler_yxz(vec3 angles, mat4 dest) {
  glm_euler_yxz(angles, dest);
}

CGLM_EXPORT
void
glmc_euler_by_order(vec3 angles, glm_euler_seq axis, mat4 dest) {
  glm_euler_by_order(angles, axis, dest);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
void
glmc_mat4_ucopy(mat4 mat, mat4 dest) {
  glm_mat4_copy(mat, dest);
}

CGLM_EXPORT
void
glmc_mat4_copy(mat4 mat, mat4 dest) {
  glm_mat4_copy(mat, dest);
}

CGLM_EXPORT
void
glmc_mat4_identity(mat4 mat) {
  glm_mat4_identity(mat);
}

CGLM_EXPORT
void
glmc_mat4_identity_array(mat4 * __restrict mat, size_t count) {
  glm_mat4_identity_array(mat, count);
}

CGLM_EXPORT
void
glmc_mat4_zero(mat4 mat) {
  glm_mat4_zero(mat);
}

CGLM_EXPORT
void
glmc_mat4_pick3(mat4 mat, mat3 dest) {
  glm_mat4_pick3(mat, dest);
}

CGLM_EXPORT
void
glmc_mat4_pick3t(mat4 mat, mat3 dest) {
  glm_mat4_pick3t(mat, dest);
}

CGLM_EXPORT
void
glmc_mat4_ins3(mat3 mat, mat4 dest) {
  glm_mat4_ins3(mat, dest);
}

CGLM_EXPORT
void
glmc_mat4_mul(mat4 m1, mat4 m2, mat4 dest) {
  glm_mat4_mul(m1, m2, dest);
}

CGLM_EXPORT
void
glmc_mat4_mulN(mat4 * __restrict matrices[], uint32_t len, mat4 dest) {
  glm_mat4_mulN(matrices, len, dest);
}

CGLM_EXPORT
void
glmc_mat4_mulv(mat4 m, vec4 v, vec4 dest) {
  glm_mat4_mulv(m, v, dest);
}

CGLM_EXPORT
void
glmc_mat4_mulv3(mat4 m, vec3 v, float last, vec3 dest) {
  glm_mat4_mulv3(m, v, last, dest);
}

CGLM_EXPORT
float
glmc_mat4_trace(mat4 m) {
  return glm_mat4_trace(m);
}

CGLM_EXPORT
float
glmc_mat4_trace3(mat4 m) {
  return glm_mat4_trace3(m);
}

CGLM_EXPORT
void
glmc_mat4_quat(mat4 m, versor dest) {
  glm_mat4_quat(m, dest);
}

CGLM_EXPORT
void
glmc_mat4_transpose_to(mat4 m, mat4 dest) {
  glm_mat4_transpose_to(m, dest);
}

CGLM_EXPORT
void
glmc_mat4_transpose(mat4 m) {
  glm_mat4_transpose(m);
}

CGLM_EXPORT
void
glmc_mat4_scale_p(mat4 m, float s) {
  glm_mat4_scale_p(m, s);
}

CGLM_EXPORT
void
glmc_mat4_scale(mat4 m, float s) {
  glm_mat4_scale(m, s);
}

CGLM_EXPORT
float
glmc_mat4_det(mat4 mat) {
  return glm_mat4_det(mat);
}

CGLM_EXPORT
void
glmc_mat4_inv(mat4 mat, mat4 dest) {
  glm_mat4_inv(mat, dest);
}

CGLM_EXPORT
void
glmc_mat4_inv_precise(mat4 mat, mat4 dest) {
  glm_mat4_inv_precise(mat, dest);
}

CGLM_EXPORT
void
glmc_mat4_inv_fast(mat4 mat, mat4 dest) {
  glm_mat4_inv_fast(mat, dest);
}

CGLM_EXPORT
void
glmc_mat4_swap_col(mat4 mat, int col1, int col2) {
  glm_mat4_swap_col(mat, col1, col2);
}

CGLM_EXPORT
void
glmc_mat4_swap_row(mat4 mat, int row1, int row2) {
  glm_mat4_swap_row(mat, row1, row2);
}

CGLM_EXPORT
float
glmc_mat4_rmc(vec4 r, mat4 m, vec4 c) {
  return glm_mat4_rmc(r, m, c);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
void
glmc_aabb_transform(vec3 box[2], mat4 m, vec3 dest[2]) {
  glm_aabb_transform(box, m, dest);
}

CGLM_EXPORT
void
glmc_aabb_merge(vec3 box1[2], vec3 box2[2], vec3 dest[2]) {
  glm_aabb_merge(box1, box2, dest);
}

CGLM_EXPORT
void
glmc_aabb_crop(vec3 box[2], vec3 cropBox[2], vec3 dest[2]) {
  glm_aabb_crop(box, cropBox, dest);
}

CGLM_EXPORT
void
glmc_aabb_crop_until(vec3 box[2],
                     vec3 cropBox[2],
                     vec3 clampBox[2],
                     vec3 dest[2]) {
  glm_aabb_crop_until(box, cropBox, clampBox, dest);
}

CGLM_EXPORT
bool
glmc_aabb_frustum(vec3 box[2], vec4 planes[6]) {
  return glm_aabb_frustum(box, planes);
}

CGLM_EXPORT
void
glmc_aabb_invalidate(vec3 box[2]) {
  glm_aabb_invalidate(box);
}

CGLM_EXPORT
bool
glmc_aabb_isvalid(vec3 box[2]) {
  return glm_aabb_isvalid(box);
}

CGLM_EXPORT
float
glmc_aabb_size(vec3 box[2]) {
  return glm_aabb_size(box);
}

CGLM_EXPORT
float
glmc_aabb_radius(vec3 box[2]) {
  return glm_aabb_radius(box);
}

CGLM_EXPORT
void
glmc_aabb_center(vec3 box[2], vec3 dest) {
  glm_aabb_center(box, dest);
}

CGLM_EXPORT
bool
glmc_aabb_aabb(vec3 box[2], vec3 other[2]) {
  return glm_aabb_aabb(box, other);
}

CGLM_EXPORT
bool
glmc_aabb_point(vec3 box[2], vec3 point) {
  return glm_aabb_point(box, point);
}

CGLM_EXPORT
bool
glmc_aabb_contains(vec3 box[2], vec3 other[2]) {
  return glm_aabb_contains(box, other);
}

CGLM_EXPORT
bool
glmc_aabb_sphere(vec3 box[2], vec4 s) {
  return glm_aabb_sphere(box, s);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
void
glmc_vec2(float * __restrict v, vec2 dest) {
  glm_vec2(v, dest);
}

CGLM_EXPORT
void
glmc_vec2_copy(vec2 a, vec2 dest) {
  glm_vec2_copy(a, dest);
}

CGLM_EXPORT
void
glmc_vec2_zero(vec2 v) {
  glm_vec2_zero(v);
}

CGLM_EXPORT
void
glmc_vec2_one(vec2 v) {
  glm_vec2_one(v);
}

CGLM_EXPORT
float
glmc_vec2_dot(vec2 a, vec2 b) {
  return glm_vec2_dot(a, b);
}

CGLM_EXPORT
float
glmc_vec2_cross(vec2 a, vec2 b) {
  return glm_vec2_cross(a, b);
}

CGLM_EXPORT
float
glmc_vec2_norm2(vec2 v) {
  return glm_vec2_norm2(v);
}

CGLM_EXPORT
float
glmc_vec2_norm(vec2 v) {
  return glm_vec2_norm(v);
}

CGLM_EXPORT
void
glmc_vec2_add(vec2 a, vec2 b, vec2 dest) {
  glm_vec2_add(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec2_adds(vec2 v, float s, vec2 dest) {
  glm_vec2_adds(v, s, dest);
}

CGLM_EXPORT
void
glmc_vec2_sub(vec2 a, vec2 b, vec2 dest) {
  glm_vec2_sub(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec2_subs(vec2 v, float s, vec2 dest) {
  glm_vec2_subs(v, s, dest);
}

CGLM_EXPORT
void
glmc_vec2_mul(vec2 a, vec2 b, vec2 dest) {
  glm_vec2_mul(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec2_scale(vec2 v, float s, vec2 dest) {
  glm_vec2_scale(v, s, dest);
}

CGLM_EXPORT
void
glmc_vec2_scale_as(vec2 v, float s, vec2 dest) {
  glm_vec2_scale_as(v, s, dest);
}

CGLM_EXPORT
void
glmc_vec2_div(vec2 a, vec2 b, vec2 dest) {
  glm_vec2_div(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec2_divs(vec2 v, float s, vec2 dest) {
  glm_vec2_divs(v, s, dest);
}

CGLM_EXPORT
void
glmc_vec2_addadd(vec2 a, vec2 b, vec2 dest) {
  glm_vec2_addadd(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec2_subadd(vec2 a, vec2 b, vec2 dest) {
  glm_vec2_subadd(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec2_muladd(vec2 a, vec2 b, vec2 dest) {
  glm_vec2_muladd(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec2_muladds(vec2 a, float s, vec2 dest) {
  glm_vec2_muladds(a, s, dest);
}

CGLM_EXPORT
void
glmc_vec2_maxadd(vec2 a, vec2 b, vec2 dest) {
  glm_vec2_maxadd(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec2_minadd(vec2 a, vec2 b, vec2 dest) {
  glm_vec2_minadd(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec2_negate_to(vec2 v, vec2 dest) {
  glm_vec2_negate_to(v, dest);
}

CGLM_EXPORT
void
glmc_vec2_negate(vec2 v) {
  glm_vec2_negate(v);
}

CGLM_EXPORT
void
glmc_vec2_normalize(vec2 v) {
  glm_vec2_normalize(v);
}

CGLM_EXPORT
void
glmc_vec2_normalize_to(vec2 v, vec2 dest) {
  glm_vec2_normalize_to(v, dest);
}

CGLM_EXPORT
void
glmc_vec2_rotate(vec2 v, float angle, vec2 dest) {
  glm_vec2_rotate(v, angle, dest);
}

CGLM_EXPORT
float
glmc_vec2_distance2(vec2 a, vec2 b) {
  return glm_vec2_distance2(a, b);
}

CGLM_EXPORT
float
glmc_vec2_distance(vec2 a, vec2 b) {
  return glm_vec2_distance(a, b);
}

CGLM_EXPORT
void
glmc_vec2_maxv(vec2 a, vec2 b, vec2 dest) {
  glm_vec2_maxv(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec2_minv(vec2 a, vec2 b, vec2 dest) {
  glm_vec2_minv(a, b, dest);
}

CGLM_EXPORT
void
glmc_vec2_clamp(vec2 v, float minval, float maxval) {
  glm_vec2_clamp(v, minval, maxval);
}

CGLM_EXPORT
void
glmc_vec2_lerp(vec2 from, vec2 to, float t, vec2 dest) {
  glm_vec2_lerp(from, to, t, dest);
}

// This empty file is needed to trick swiftpm to build the header-only version of cglm as swiftpm itself does not support C targets that have no source code files
/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
void
glmc_frustum(float left,
             float right,
             float bottom,
             float top,
             float nearVal,
             float farVal,
             mat4 dest) {
  glm_frustum(left,
              right,
              bottom,
              top,
              nearVal,
              farVal,
              dest);
}

CGLM_EXPORT
void
glmc_ortho(float left,
           float right,
           float bottom,
           float top,
           float nearVal,
           float farVal,
           mat4 dest) {
  glm_ortho(left,
            right,
            bottom,
            top,
            nearVal,
            farVal,
            dest);
}

CGLM_EXPORT
void
glmc_ortho_aabb(vec3 box[2], mat4 dest) {
  glm_ortho_aabb(box, dest);
}

CGLM_EXPORT
void
glmc_ortho_aabb_p(vec3 box[2], float padding, mat4 dest) {
  glm_ortho_aabb_p(box, padding, dest);
}

CGLM_EXPORT
void
glmc_ortho_aabb_pz(vec3 box[2], float padding, mat4 dest) {
  glm_ortho_aabb_pz(box, padding, dest);
}

CGLM_EXPORT
void
glmc_ortho_default(float aspect, mat4 dest) {
  glm_ortho_default(aspect, dest);
}

CGLM_EXPORT
void
glmc_ortho_default_s(float aspect, float size, mat4 dest) {
  glm_ortho_default_s(aspect, size, dest);
}

CGLM_EXPORT
void
glmc_perspective(float fovy,
                 float aspect,
                 float nearVal,
                 float farVal,
                 mat4 dest) {
  glm_perspective(fovy,
                  aspect,
                  nearVal,
                  farVal,
                  dest);
}

CGLM_EXPORT
void
glmc_persp_move_far(mat4 proj, float deltaFar) {
  glm_persp_move_far(proj, deltaFar);
}

CGLM_EXPORT
void
glmc_perspective_default(float aspect, mat4 dest) {
  glm_perspective_default(aspect, dest);
}

CGLM_EXPORT
void
glmc_perspective_resize(float aspect, mat4 proj) {
  glm_perspective_resize(aspect, proj);
}

CGLM_EXPORT
void
glmc_lookat(vec3 eye,
            vec3 center,
            vec3 up,
            mat4 dest) {
  glm_lookat(eye, center, up, dest);
}

CGLM_EXPORT
void
glmc_look(vec3 eye, vec3 dir, vec3 up, mat4 dest) {
  glm_look(eye, dir, up, dest);
}

CGLM_EXPORT
void
glmc_look_anyup(vec3 eye, vec3 dir, mat4 dest) {
  glm_look_anyup(eye, dir, dest);
}

CGLM_EXPORT
void
glmc_persp_decomp(mat4 proj,
                  float * __restrict nearVal,
                  float * __restrict farVal,
                  float * __restrict top,
                  float * __restrict bottom,
                  float * __restrict left,
                  float * __restrict right) {
  glm_persp_decomp(proj, nearVal, farVal, top, bottom, left, right);
}

CGLM_EXPORT
void
glmc_persp_decompv(mat4 proj, float dest[6]) {
  glm_persp_decompv(proj, dest);
}

CGLM_EXPORT
void
glmc_persp_decomp_x(mat4 proj,
                    float * __restrict left,
                    float * __restrict right) {
  glm_persp_decomp_x(proj, left, right);
}

CGLM_EXPORT
void
glmc_persp_decomp_y(mat4 proj,
                    float * __restrict top,
                    float * __restrict bottom) {
  glm_persp_decomp_y(proj, top, bottom);
}

CGLM_EXPORT
void
glmc_persp_decomp_z(mat4 proj,
                    float * __restrict nearVal,
                    float * __restrict farVal) {
  glm_persp_decomp_z(proj, nearVal, farVal);
}

CGLM_EXPORT
void
glmc_persp_decomp_far(mat4 proj, float * __restrict farVal) {
  glm_persp_decomp_far(proj, farVal);
}

CGLM_EXPORT
void
glmc_persp_decomp_near(mat4 proj, float * __restrict nearVal) {
  glm_persp_decomp_near(proj, nearVal);
}

CGLM_EXPORT
float
glmc_persp_fovy(mat4 proj) {
  return glm_persp_fovy(proj);
}

CGLM_EXPORT
float
glmc_persp_aspect(mat4 proj) {
  return glm_persp_aspect(proj);
}

CGLM_EXPORT
void
glmc_persp_sizes(mat4 proj, float fovy, vec4 dest) {
  glm_persp_sizes(proj, fovy, dest);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
float
glmc_bezier(float s, float p0, float c0, float c1, float p1) {
  return glm_bezier(s, p0, c0, c1, p1);
}

CGLM_EXPORT
float
glmc_hermite(float s, float p0, float t0, float t1, float p1) {
  return glm_hermite(s, p0, t0, t1, p1);
}

CGLM_EXPORT
float
glmc_decasteljau(float prm, float p0, float c0, float c1, float p1) {
  return glm_decasteljau(prm, p0, c0, c1, p1);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
float
glmc_sphere_radii(vec4 s) {
  return glm_sphere_radii(s);
}

CGLM_EXPORT
void
glmc_sphere_transform(vec4 s, mat4 m, vec4 dest) {
  glm_sphere_transform(s, m, dest);
}

CGLM_EXPORT
void
glmc_sphere_merge(vec4 s1, vec4 s2, vec4 dest) {
  glm_sphere_merge(s1, s2, dest);
}

CGLM_EXPORT
bool
glmc_sphere_sphere(vec4 s1, vec4 s2) {
  return glm_sphere_sphere(s1, s2);
}

CGLM_EXPORT
bool
glmc_sphere_point(vec4 s, vec3 point) {
  return glm_sphere_point(s, point);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
void
glmc_frustum_planes(mat4 m, vec4 dest[6]) {
  glm_frustum_planes(m, dest);
}

CGLM_EXPORT
void
glmc_frustum_corners(mat4 invMat, vec4 dest[8]) {
  glm_frustum_corners(invMat, dest);
}

CGLM_EXPORT
void
glmc_frustum_center(vec4 corners[8], vec4 dest) {
  glm_frustum_center(corners, dest);
}

CGLM_EXPORT
void
glmc_frustum_box(vec4 corners[8], mat4 m, vec3 box[2]) {
  glm_frustum_box(corners, m, box);
}

CGLM_EXPORT
void
glmc_frustum_corners_at(vec4  corners[8],
                        float splitDist,
                        float farDist,
                        vec4  planeCorners[4]) {
  glm_frustum_corners_at(corners, splitDist, farDist, planeCorners);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
void
glmc_mat3_copy(mat3 mat, mat3 dest) {
  glm_mat3_copy(mat, dest);
}

CGLM_EXPORT
void
glmc_mat3_identity(mat3 mat) {
  glm_mat3_identity(mat);
}

CGLM_EXPORT
void
glmc_mat3_zero(mat3 mat) {
  glm_mat3_zero(mat);
}

CGLM_EXPORT
void
glmc_mat3_identity_array(mat3 * __restrict mat, size_t count) {
  glm_mat3_identity_array(mat, count);
}

CGLM_EXPORT
void
glmc_mat3_mul(mat3 m1, mat3 m2, mat3 dest) {
  glm_mat3_mul(m1, m2, dest);
}

CGLM_EXPORT
void
glmc_mat3_transpose_to(mat3 m, mat3 dest) {
  glm_mat3_transpose_to(m, dest);
}

CGLM_EXPORT
void
glmc_mat3_transpose(mat3 m) {
  glm_mat3_transpose(m);
}

CGLM_EXPORT
void
glmc_mat3_mulv(mat3 m, vec3 v, vec3 dest) {
  glm_mat3_mulv(m, v, dest);
}

CGLM_EXPORT
float
glmc_mat3_trace(mat3 m) {
  return glm_mat3_trace(m);
}

CGLM_EXPORT
void
glmc_mat3_quat(mat3 m, versor dest) {
  glm_mat3_quat(m, dest);
}

CGLM_EXPORT
void
glmc_mat3_scale(mat3 m, float s) {
  glm_mat3_scale(m, s);
}

CGLM_EXPORT
float
glmc_mat3_det(mat3 mat) {
  return glm_mat3_det(mat);
}

CGLM_EXPORT
void
glmc_mat3_inv(mat3 mat, mat3 dest) {
  glm_mat3_inv(mat, dest);
}

CGLM_EXPORT
void
glmc_mat3_swap_col(mat3 mat, int col1, int col2) {
  glm_mat3_swap_col(mat, col1, col2);
}

CGLM_EXPORT
void
glmc_mat3_swap_row(mat3 mat, int row1, int row2) {
  glm_mat3_swap_row(mat, row1, row2);
}

CGLM_EXPORT
float
glmc_mat3_rmc(vec3 r, mat3 m, vec3 c) {
  return glm_mat3_rmc(r, m, c);
}

/*
 * Copyright (c), Recep Aslantas.
 *
 * MIT License (MIT), http://opensource.org/licenses/MIT
 * Full license can be found in the LICENSE file
 */


CGLM_EXPORT
void
glmc_plane_normalize(vec4 plane) {
  glm_plane_normalize(plane);
}

