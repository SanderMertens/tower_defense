#define FLECS_COMPONENTS_GRAPHICS_IMPL

#include "flecs_components_graphics.h"

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

void FlecsComponentsGraphicsImport(
    ecs_world_t *world)
{
    ECS_MODULE(world, FlecsComponentsGraphics);
    ECS_IMPORT(world, FlecsComponentsCglm);

    ecs_set_name_prefix(world, "Ecs");

    /* Preregister before injecting metadata, so we can register ctor */
    // ECS_COMPONENT_DEFINE(world, EcsCamera);
    // ecs_set_component_actions(world, EcsCamera, {
    //     .ctor = ecs_ctor(EcsCamera)
    // });

    ECS_META_COMPONENT(world, EcsCamera);
    ECS_META_COMPONENT(world, EcsDirectionalLight);
    ECS_META_COMPONENT(world, EcsRgb);
    ECS_META_COMPONENT(world, EcsRgba);
    ECS_META_COMPONENT(world, EcsSpecular);
    ECS_META_COMPONENT(world, EcsEmissive);
}

