#define FLECS_COMPONENTS_TRANSFORM_IMPL

#include "flecs_components_transform.h"

ECS_COMPONENT_DECLARE(EcsTransform2);
ECS_COMPONENT_DECLARE(EcsTransform3);
ECS_COMPONENT_DECLARE(EcsProject2);
ECS_COMPONENT_DECLARE(EcsProject3);

void FlecsComponentsTransformImport(
    ecs_world_t *world)
{
    ECS_MODULE(world, FlecsComponentsTransform);
    ECS_IMPORT(world, FlecsComponentsCglm);

    ecs_set_name_prefix(world, "Ecs");

    ECS_META_COMPONENT(world, EcsPosition2);
    ECS_META_COMPONENT(world, EcsPosition3);
    ECS_META_COMPONENT(world, EcsScale2);
    ECS_META_COMPONENT(world, EcsScale3);
    ECS_META_COMPONENT(world, EcsRotation2);
    ECS_META_COMPONENT(world, EcsRotation3);
    ECS_META_COMPONENT(world, EcsQuaternion);

    ECS_COMPONENT_DEFINE(world, EcsTransform2);
    ECS_COMPONENT_DEFINE(world, EcsTransform3);
    ECS_COMPONENT_DEFINE(world, EcsProject2);
    ECS_COMPONENT_DEFINE(world, EcsProject3);

    ecs_set_hooks(world, EcsTransform2, {
        .ctor = flecs_default_ctor
    });

    ecs_set_hooks(world, EcsTransform3, {
        .ctor = flecs_default_ctor
    });
}

