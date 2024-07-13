#include "etc/sokol/shaders/common.glsl"

uniform vec3 u_light_ambient;
uniform vec3 u_light_ambient_ground;
uniform vec3 u_light_direction;
uniform vec3 u_light_color;
uniform vec3 u_eye_pos;
uniform float u_light_ambient_ground_falloff;
uniform float u_light_ambient_ground_offset;
uniform float u_light_ambient_ground_intensity;
uniform float u_shadow_map_size;
uniform sampler2D shadow_map;
uniform float u_shadow_far;

#define MAX_LIGHT_COUNT 64
uniform vec3 u_point_light_color[MAX_LIGHT_COUNT];
uniform vec3 u_point_light_position[MAX_LIGHT_COUNT];
uniform float u_point_light_distance[MAX_LIGHT_COUNT];

uniform int u_light_count;

in vec4 position;
in vec4 light_position;
in vec3 normal;
in vec4 color;
in vec3 material;
out vec4 frag_color;

const int pcf_count = 1;
const int pcf_samples = (2 * pcf_count + 1) * (2 * pcf_count + 1);
const float texel_c = 1.0;

float sampleShadow(sampler2D shadowMap, vec2 uv, float compare, float bias) {
  float depth = rgba_to_float(texture(shadowMap, vec2(uv.x, uv.y)));
  depth += bias;
  return step(compare, depth);
}

float sampleShadowPCF(sampler2D shadowMap, vec2 uv, float texel_size, float compare, float n_dot_l, float d) {
  float result = 0.0;
  float cos_theta = clamp(n_dot_l, 0.0, 1.0);
  float bias = 0.005;

  if (uv.x < 0. || uv.x > 1.) {
    return 1.0;
  }
  if (uv.y < 0. || uv.y > 1.) {
    return 1.0;
  }

  for (int x = -pcf_count; x <= pcf_count; x++) {
      for (int y = -pcf_count; y <= pcf_count; y++) {
          result += sampleShadow(
            shadowMap, uv + vec2(x, y) * texel_size * texel_c, compare, bias);
      }
  }

  result /= float(pcf_samples);

  return result;
}

vec3 applyAmbientGround(vec3 n) {
  if (u_light_ambient_ground_falloff > 0.0) {
    vec3 l = vec3(0.0, -1.0, 0.0);
    float n_dot_l = dot(n, l);
    float n_dot_l_inv = 1.0 - n_dot_l;
    float falloff = (u_light_ambient_ground_falloff - position.y) / u_light_ambient_ground_falloff;
    vec3 sideLight = exp2(falloff) * n_dot_l_inv * u_light_ambient_ground;
    vec3 floorLight = 0.1 * exp2(falloff) * n_dot_l * u_light_ambient_ground;
    if (position.y < u_light_ambient_ground_offset) {
      sideLight *= 0.1;
    }
    return (sideLight + floorLight) * u_light_ambient_ground_intensity;
  } else {
    return vec3(0.0, 0.0, 0.0);
  }
}

vec3 applyLight(vec3 n, vec3 v, float shininess, vec3 pos, vec3 color, float maxDistance) {
  vec3 lightPos = pos;
  vec3 eyePos = u_eye_pos;
  eyePos[0] *= -1.0;
  lightPos += u_eye_pos;
  vec3 l = position.xyz - lightPos;

  float n_dot_l = dot(n, l);
  if (n_dot_l >= 0.0) {
    float distance = length(position.xyz - lightPos);
    float attenuation = exp(-distance / maxDistance);

    // Specular
    vec3 r = reflect(normalize(l), n);
    float r_dot_v = max(dot(r, v), 0.0);
    float l_shiny = pow(r_dot_v * n_dot_l, shininess);
    vec3 l_specular = vec3(material.x * l_shiny * color);

    return color * attenuation * n_dot_l + l_specular * attenuation;
  } else {
    return vec3(0.0, 0.0, 0.0);
  }
}

vec3 applyLights(vec3 n, vec3 v, float shininess) {
  int i;

  vec3 result = vec3(0.0, 0.0, 0.0);
  for (i = 0; i < u_light_count; i ++) {
    result += applyLight(n, v, shininess,
      u_point_light_position[i], 
      u_point_light_color[i],
      u_point_light_distance[i]);
  }

  return result;
}

void main() {
  float specular_power = material.x;
  float shininess = max(material.y, 1.0);
  float emissive = material.z;
  vec3 l = normalize(u_light_direction);
  vec3 n = normalize(normal);
  float n_dot_l = dot(n, l);

  vec3 vd = u_eye_pos - position.xyz;
  vec3 v = normalize(vd);

  if (n_dot_l >= 0.0) {
    vec3 r = reflect(l, n);
    
    // Shadows
    vec3 light_pos = light_position.xyz / light_position.w;
    vec2 sm_uv = (light_pos.xy + 1.0) * 0.5;
    float depth = light_position.z;
    float texel_size = 1.0 / u_shadow_map_size;
    float d = length(vd);
    float s = sampleShadowPCF(shadow_map, sm_uv, texel_size, depth, n_dot_l, d);
    s = max(s, emissive);

    float r_dot_v = max(dot(r, v), 0.0);
    float l_shiny = pow(r_dot_v * n_dot_l, shininess);
    vec3 l_specular = vec3(specular_power * l_shiny * u_light_color);
    vec3 l_diffuse = vec3(u_light_color) * n_dot_l;
    vec3 l_light = (u_light_ambient + s * l_diffuse);
    frag_color = vec4(max(vec3(emissive), l_light) * color.xyz + s * l_specular, 1.0);
  } else {
    vec3 light = emissive + clamp(1.0 - emissive, 0.0, 1.0) * (u_light_ambient);
    frag_color = vec4(light * color.xyz, 1.0);
  }

  frag_color += vec4(applyLights(n, v, shininess), 0.0);
  frag_color += vec4(applyAmbientGround(n).xyz, 0.0);
}
