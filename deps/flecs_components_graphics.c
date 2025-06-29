#define FLECS_COMPONENTS_GRAPHICS_IMPL

#include "flecs_components_graphics.h"

ECS_TAG_DECLARE(EcsSun);

ECS_CTOR(EcsCamera, ptr, {
    ptr->position[0] = 0.0f;
    ptr->position[1] = 0.0f;
    ptr->position[2] = 0.0f;

    ptr->lookat[0] = 0.0f;
    ptr->lookat[1] = 1.0f;
    ptr->lookat[2] = 1.0f;

    ptr->up[0] = 0.0f;
    ptr->up[1] = -1.0f;
    ptr->up[2] = 0.0f;

    ptr->fov = 30;
    ptr->near_ = 0.1;
    ptr->far_ = 1000;
})

ECS_CTOR(EcsAtmosphere, ptr, {
    ptr->intensity = 7.0;
    ptr->planet_radius = 6371e3;
    ptr->atmosphere_radius = 6471e3;
    ptr->rayleigh_coef[0] = 5.5e-6;
    ptr->rayleigh_coef[1] = 13.0e-6;
    ptr->rayleigh_coef[2] = 22.4e-6;
    ptr->mie_coef = 21e-6;
    ptr->rayleigh_scale_height = 8e3;
    ptr->mie_scale_height = 1.2e3;
    ptr->mie_scatter_dir = 0.758;
})

static void UpdateSelfLights(ecs_iter_t *it) {
    EcsSelfLight *sl = ecs_field(it, EcsSelfLight, 0);
    EcsRgb *color = ecs_field(it, EcsRgb, 1);
    EcsEmissive *emissive = ecs_field(it, EcsEmissive, 2);
    EcsPointLight *out = ecs_field(it, EcsPointLight, 3);

    for (int i = 0; i < it->count; i ++) {
        out[i].distance = sl[i].distance;
        out[i].color[0] = color[i].r;
        out[i].color[1] = color[i].g;
        out[i].color[2] = color[i].b;
        if (emissive) {
            out[i].intensity = emissive[i].value;
        } else {
            out[i].intensity = 1.0;
        }
    }
}

void FlecsComponentsGraphicsImport(
    ecs_world_t *world)
{
    ECS_MODULE(world, FlecsComponentsGraphics);
    ECS_IMPORT(world, FlecsComponentsCglm);

    ecs_set_name_prefix(world, "Ecs");
    ECS_META_COMPONENT(world, EcsRgb);

    ecs_set_name_prefix(world, "Ecs");
    ECS_META_COMPONENT(world, EcsCamera);
    ECS_META_COMPONENT(world, EcsLookAt);
    ECS_META_COMPONENT(world, EcsDirectionalLight);
    ECS_META_COMPONENT(world, EcsPointLight);
    ECS_META_COMPONENT(world, EcsSelfLight);
    ECS_META_COMPONENT(world, EcsSpecular);
    ECS_META_COMPONENT(world, EcsEmissive);
    ECS_META_COMPONENT(world, EcsLightIntensity);
    ECS_META_COMPONENT(world, EcsAtmosphere);
    ECS_TAG_DEFINE(world, EcsSun);

    ecs_add_pair(world, ecs_id(EcsRgb), EcsOnInstantiate, EcsInherit);
    ecs_add_pair(world, ecs_id(EcsSpecular), EcsOnInstantiate, EcsInherit);
    ecs_add_pair(world, ecs_id(EcsEmissive), EcsOnInstantiate, EcsInherit);

    ecs_add_pair(world, ecs_id(EcsDirectionalLight), EcsOnInstantiate, EcsInherit);
    ecs_add_pair(world, ecs_id(EcsPointLight), EcsOnInstantiate, EcsInherit);
    ecs_add_pair(world, ecs_id(EcsSelfLight), EcsOnInstantiate, EcsInherit);

    ecs_add_pair(world, ecs_id(EcsSelfLight), EcsWith, ecs_id(EcsPointLight));

    ecs_struct(world, {
        .entity = ecs_entity(world, { 
            .name = "ecs_rgb_t",
            .symbol = "ecs_rgb_t",
        }),
        .members = {
            { .name = "r", .type = ecs_id(ecs_f32_t) },
            { .name = "g", .type = ecs_id(ecs_f32_t) },
            { .name = "b", .type = ecs_id(ecs_f32_t) }
        }
    });

    ecs_set_hooks(world, EcsAtmosphere, {
        .ctor = ecs_ctor(EcsAtmosphere)
    });

    ECS_SYSTEM(world, UpdateSelfLights, EcsPostUpdate,
        [in]  SelfLight, 
        [in]  Rgb, 
        [in]  ?Emissive,
        [out] PointLight);
}

