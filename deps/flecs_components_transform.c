#define FLECS_COMPONENTS_TRANSFORM_IMPL

#include "flecs_components_transform.h"

ECS_TAG_DECLARE(EcsTransformManually);
ECS_TAG_DECLARE(EcsTransformOnce);
ECS_TAG_DECLARE(EcsTransformNeeded);

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

    ECS_TAG_DEFINE(world, EcsTransformManually);
    ECS_TAG_DEFINE(world, EcsTransformOnce);
    ECS_TAG_DEFINE(world, EcsTransformNeeded);

    ecs_add_pair(world, EcsTransformOnce, EcsWith, EcsTransformNeeded);

    ecs_set_hooks(world, EcsPosition2, {
        .ctor = flecs_default_ctor
    });

    ecs_set_hooks(world, EcsPosition3, {
        .ctor = flecs_default_ctor
    });

    ecs_set_hooks(world, EcsScale2, {
        .ctor = flecs_default_ctor
    });

    ecs_set_hooks(world, EcsScale3, {
        .ctor = flecs_default_ctor
    });

    ecs_set_hooks(world, EcsRotation2, {
        .ctor = flecs_default_ctor
    });

    ecs_set_hooks(world, EcsRotation3, {
        .ctor = flecs_default_ctor
    });

    ecs_set_hooks(world, EcsTransform2, {
        .ctor = flecs_default_ctor
    });

    ecs_set_hooks(world, EcsTransform3, {
        .ctor = flecs_default_ctor
    });

    ecs_add_pair(world, ecs_id(EcsPosition3), EcsWith, ecs_id(EcsTransform3));
    ecs_add_pair(world, ecs_id(EcsRotation3), EcsWith, ecs_id(EcsTransform3));
    ecs_add_pair(world, ecs_id(EcsScale3),    EcsWith, ecs_id(EcsTransform3));
}

