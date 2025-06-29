#define FLECS_COMPONENTS_PHYSICS_IMPL
#include "flecs_components_physics.h"

ECS_DECLARE(EcsCollider);
ECS_DECLARE(EcsRigidBody);

void FlecsComponentsPhysicsImport(
    ecs_world_t *world)
{
    ECS_MODULE(world, FlecsComponentsPhysics);

    ecs_set_name_prefix(world, "Ecs");

    ECS_TAG_DEFINE(world, EcsCollider);
    ECS_TAG_DEFINE(world, EcsRigidBody);
    ECS_META_COMPONENT(world, EcsVelocity2);
    ECS_META_COMPONENT(world, EcsVelocity3);
    ECS_META_COMPONENT(world, EcsAngularSpeed);
    ECS_META_COMPONENT(world, EcsAngularVelocity);
    ECS_META_COMPONENT(world, EcsBounciness);
    ECS_META_COMPONENT(world, EcsFriction);

    ecs_set_hooks(world, EcsVelocity2, {
        .ctor = flecs_default_ctor
    });

    ecs_set_hooks(world, EcsVelocity3, {
        .ctor = flecs_default_ctor
    });

    ecs_set_hooks(world, EcsAngularSpeed, {
        .ctor = flecs_default_ctor
    });

    ecs_set_hooks(world, EcsAngularVelocity, {
        .ctor = flecs_default_ctor
    });

    ecs_set_hooks(world, EcsBounciness, {
        .ctor = flecs_default_ctor
    });

    ecs_set_hooks(world, EcsFriction, {
        .ctor = flecs_default_ctor
    });
}

