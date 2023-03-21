float ambientOcclusion = rgba_to_float(texture(t_occlusion, uv));
frag_color = (1.0 - ambientOcclusion) * texture(t_scene, uv);
