#include <tower_defense.h>
#include <vector>

using namespace flecs::components;

// Shortcuts to imported components
using Position = transform::Position3;
using Rotation = transform::Rotation3;
using Velocity = physics::Velocity3;
using Input = input::Input;
using SpatialQuery = flecs::systems::physics::SpatialQuery;
using SpatialQueryResult = flecs::systems::physics::SpatialQueryResult;
using Color = graphics::Color;
using Specular = graphics::Specular;
using Emissive = graphics::Emissive;
using Box = geometry::Box;

#define ECS_PI_2 ((float)(GLM_PI * 2))

// Game constants
static const float CameraAcceleration = 0.2;
static const float CameraDeceleration = 0.1;
static const float CameraMaxSpeed = 0.05;
static const float CameraDistance = 8;
static const float CameraHeight = 6.0;
static const float CameraLookatRadius = 2.0;

static const float EnemySize = 0.3;
static const float EnemySpeed = 2.5;
static const float EnemySpawnInterval = 0.01;

static const float TurretRotateSpeed = 1.5;
static const float TurretFireInterval = 0.1;
static const float TurretRange = 3.0;
static const float TurretCannonOffset = 0.1;
static const float TurretCannonLength = 0.3;

static const float BulletSize = 0.05;
static const float BulletSpeed = 12.0;
static const float BulletLifespan = 0.5;
static const float BulletDamage = 0.03;

static const float BeamDamage = 0.07;

static const float ShellSize = 0.03;
static const float ShellSpeed = 0.05;
static const float ShellLifespan = 0.3;

static const float FireballSize = 0.15;
static const float FireballSizeDecay = 0.8;
static const float FireballLifespan = 0.1;

static const float BoltSize = 0.4;
static const float BoltSizeDecay = 0.9;
static const float BoltLifespan = 5.3;

static const float SmokeSize = 0.4;
static const float SmokeSizeDecay = 0.98;
static const float SmokeColorDecay = 0.9;
static const float SmokeLifespan = 5.0;

static const float SparkSize = 0.15;
static const float SparkLifespan = 0.5;
static const float SparkSizeDecay = 0.9;
static const float SparkVelocityDecay = 0.9;

static const float TileSize = 1.5;
static const float TileHeight = 0.4;
static const float PathHeight = 0.1;
static const float TileSpacing = 0.0;
static const int TileCountX = 10;
static const int TileCountZ = 10;

// Direction vector. During pathfinding enemies will cycle through this vector
// to find the next direction to turn to.
static const transform::Position2 dir[] = {
    {-1, 0},
    {0, -1},
    {1, 0},
    {0, 1},
};

// Grid wrapper around a vector which translates an (x, y) into a vector index
template <typename T>
class grid {
public:
    grid(int width, int height)
        : m_width(width)
        , m_height(height) 
    { 
        for (int x = 0; x < width; x ++) {
            for (int y = 0; y < height; y ++) {
                m_values.push_back(T());
            }
        }
    }

    void set(int32_t x, int32_t y, T value) {
        m_values[y * m_width + x] = value;
    }

    const T operator()(int32_t x, int32_t y) {
        return m_values[y * m_width + x];
    }

private:
    int m_width;
    int m_height;
    std::vector<T> m_values;
};

// Global game data
struct Game {
    flecs::entity tile_prefab;
    flecs::entity path_prefab;
    flecs::entity enemy_prefab;
    flecs::entity cannon_prefab;
    flecs::entity railgun_prefab;
    flecs::entity bullet_prefab;
    flecs::entity beam_prefab;
    flecs::entity shell_prefab;
    flecs::entity fireball_prefab;
    flecs::entity bolt_prefab;
    flecs::entity smoke_prefab;
    flecs::entity spark_prefab;
    flecs::entity window;
    flecs::entity level;
    
    Position center;
    float size;        
};

// Level map and spawn point for enemies
struct Level {
    Level() {
        map = nullptr;
    }

    Level(grid<bool> *arg_map, transform::Position2 arg_spawn) {
        map = arg_map;
        spawn_point = arg_spawn;
    }

    grid<bool> *map;
    transform::Position2 spawn_point;  
};

// Particles
struct Particle {
    float size_decay;
    float color_decay;
    float velocity_decay;
    float lifespan;
};

struct ParticleLifespan {
    ParticleLifespan() {
        t = 0;
    }
    float t;
};

// Enemy components
struct Enemy { };

struct Direction {
    int value;
};

struct Health {
    Health() {
        value = 1;
    }
    float value;
};

// Bullet components
struct Bullet { };

// Turret components
struct Turret { 
    Turret(float fire_interval_arg = 1.0) {
        lr = 1;
        t_since_fire = 0;
        fire_interval = fire_interval_arg;
        beam_countdown = 0;
    }

    float fire_interval;
    float t_since_fire;
    float beam_countdown;
    flecs::entity head;
    flecs::entity beam;
    int lr;
};

struct Cannon { };

struct Railgun { };

struct Target {
    Target() {
        prev_position[0] = 0;
        prev_position[1] = 0;
        prev_position[2] = 0;
        lock = false;
    }

    flecs::entity target;
    vec3 prev_position;
    vec3 aim_position;
    float angle;
    float distance;
    bool lock;
};

// Camera components
struct CameraController {
    CameraController(float ar = 0, float av = 0, float av_h = 0, 
        float ah = CameraHeight, float ad = CameraDistance, float as = 0)
    {
        r = ar;
        v = av;
        v_h = av_h;
        h = ah;
        d = ad;
        shake = as;
    }

    float r;
    float v;
    float v_h;
    float h;
    float d;
    float shake;
};

// Utility functions

float randf(float scale) {
    return ((float)rand() / (float)RAND_MAX) * scale;
}

float to_coord(float x) {
    return x * (TileSpacing + TileSize) - (TileSize / 2.0);
}

float from_coord(float x) {
    return (x + (TileSize / 2.0)) / (TileSpacing + TileSize);
}

float to_x(float x) {
    return to_coord(x + 0.5) - to_coord((TileCountX / 2.0));
}

float to_z(float z) {
    return to_coord(z);
}

float from_x(float x) {
    return from_coord(x + to_coord((TileCountX / 2.0))) - 0.5;
}

float from_z(float z) {
    return from_coord(z);
}

float angle_normalize(float angle) {
    return angle - floor(angle / ECS_PI_2) * ECS_PI_2;
}

float look_at(vec3 eye, vec3 dest) {
    vec3 diff;
    
    glm_vec3_sub(dest, eye, diff);
    float x = fabs(diff[0]), z = fabs(diff[2]);
    bool x_sign = diff[0] < 0, z_sign = diff[2] < 0;
    float r = atan(z / x);

    if (z_sign) {
        r += GLM_PI;
    }

    if (z_sign == x_sign) {
        r = -r + GLM_PI;
    }

    return angle_normalize(r + GLM_PI);
}

float rotate_to(float cur, float target, float increment) {
    cur = angle_normalize(cur);
    target = angle_normalize(target);

    if (cur - target > GLM_PI) {
        cur -= ECS_PI_2;
    } else if (cur - target < -GLM_PI) {
        cur += ECS_PI_2;
    }
    
    if (cur > target) {
        cur -= increment;
        if (cur < target) {
            cur = target;
        }
    } else {
        cur += increment;
        if (cur > target) {
            cur = target;
        }
    }

    return cur;
}

// Check if enemy needs to change direction
bool find_path(Position& p, Direction& d, const Level* lvl) {
    // Check if enemy is in center of tile
    float t_x = from_x(p.x);
    float t_y = from_z(p.z);
    int ti_x = (int)t_x;
    int ti_y = (int)t_y;
    float td_x = t_x - ti_x;
    float td_y = t_y - ti_y;

    // If enemy is in center of tile, decide where to go next
    if (td_x < 0.1 && td_y < 0.1) {
        grid<bool> *tiles = lvl->map;

        // Compute backwards direction so we won't try to go there
        int backwards = (d.value + 2) % 4;

        // Find a direction that the enemy can move to
        for (int i = 0; i < 3; i ++) {
            int n_x = ti_x + dir[d.value].x;
            int n_y = ti_y + dir[d.value].y;

            if (n_x >= 0 && n_x <= TileCountX) {
                if (n_y >= 0 && n_y <= TileCountZ) {
                    // Next tile is still on the grid, test if it's a path
                    if (tiles[0](n_x, n_y)) {
                        // Next tile is a path, so continue along current direction
                        return false;
                    }
                }
            }

            // Try next direction. Make sure not to move backwards
            do {
                d.value = (d.value + 1) % 4;
            } while (d.value == backwards);
        }

        // If enemy was not able to find a next direction, it reached the end
        return true;        
    }

    return false;
}

float decelerate_camera(float v, float delta_time, float max_speed) {
    if (v > 0) {
        v = glm_clamp(v - CameraDeceleration * delta_time, 0, v);
    }
    if (v < 0) {
        v = glm_clamp(v + CameraDeceleration * delta_time, v, 0);
    }

    return glm_clamp(v, -max_speed, max_speed);
}

// Move camera around with keyboard
void MoveCamera(flecs::iter& it, CameraController *ctrl) {
    auto input = it.column<const Input>(2);
    auto camera = it.column<graphics::Camera>(3);

    // Accelerate camera if keys are pressed
    if (input->keys[ECS_KEY_D].state) {
        ctrl->v -= CameraAcceleration * it.delta_time();
    }
    if (input->keys[ECS_KEY_A].state) {
        ctrl->v += CameraAcceleration * it.delta_time();
    }  
    if (input->keys[ECS_KEY_S].state) {
        ctrl->v_h -= CameraAcceleration * it.delta_time();
    }
    if (input->keys[ECS_KEY_W].state) {
        ctrl->v_h += CameraAcceleration * it.delta_time();
    }

    float max_speed = CameraMaxSpeed;
    if (input->keys[ECS_KEY_SHIFT].state) {
        max_speed /= 8;
    }

    // Decelerate camera each frame
    ctrl->v = decelerate_camera(ctrl->v, it.delta_time(), max_speed);
    ctrl->v_h = decelerate_camera(ctrl->v_h, it.delta_time(), max_speed);

    // Update camera spherical coordinates
    ctrl->r += ctrl->v;
    ctrl->h += ctrl->v_h * 2;
    ctrl->d -= ctrl->v_h;

    camera->position[0] = cos(ctrl->r) * ctrl->d;
    camera->position[1] = ctrl->h;
    camera->position[2] = sin(ctrl->r) * ctrl->d + to_z(TileCountZ / 2);

    // Camera shake
    camera->position[1] += sin(it.world_time() * 50) * ctrl->shake;
    ctrl->shake *= 0.80;

    camera->lookat[0] = cos(ctrl->r) * CameraLookatRadius;
    camera->lookat[2] = sin(ctrl->r) * CameraLookatRadius + to_z(TileCountZ / 2);

    if (input->keys[ECS_KEY_MINUS].state) {
        it.world().set_time_scale(it.world().get_time_scale() * 0.95);
    }

    if (input->keys[ECS_KEY_PLUS].state) {
        it.world().set_time_scale(it.world().get_time_scale() * 1.05);
    }
}

// Periodically spawn new enemies
void SpawnEnemy(flecs::iter& it) {
    auto g = it.column<const Game>(1);
    const Level* lvl = g->level.get<Level>();

    it.world().entity().add_instanceof(g->enemy_prefab)
        .set<Direction>({0})
        .set<Position>({
            lvl->spawn_point.x, 0.6, lvl->spawn_point.y
        });
}

// Progress enemies along path
void MoveEnemy(flecs::iter& it, Position* p, Direction* d) {
    auto g = it.column<const Game>(3);
    const Level* lvl = g->level.get<Level>();

    for (int i = 0; i < it.count(); i ++) {
        if (find_path(p[i], d[i], lvl)) {
            it.entity(i).destruct();
        } else {
            p[i].x += dir[d[i].value].x * EnemySpeed * it.delta_time();
            p[i].z += dir[d[i].value].y * EnemySpeed * it.delta_time();
        }
    }
}

// Clear target if it has been destroyed or if it is out of range
void ClearTarget(flecs::iter& it, Target* target, Position* p) {
    for (auto i : it) {
        auto t = target[i].target;
        if (t) {
            if (!t.is_alive()) {
                target[i].target = flecs::entity::null();
                target[i].lock = false;
            } else {
                const Position *tp = t.get<Position>(); // TODO: look into lifecycle issue
                if (!tp) {
                    continue;
                }

                Position target_pos = *tp;
                float distance = glm_vec3_distance(p[i], target_pos);
                if (distance > TurretRange) {
                    target[i].target = flecs::entity::null();
                    target[i].lock = false;
                }
            }
        }
    }
}

// Find target for turret
void FindTarget(flecs::iter& it, Target* target, Position* p) {
    auto q = it.column<const SpatialQuery>(3);
    auto qr = it.column<SpatialQueryResult>(4);

    for (auto i : it) {
        if (target[i].target) {
            continue;
        }

        flecs::entity enemy;
        float distance = 0, min_distance = 0;
        q->query.findn(p[i], TurretRange, qr[i].results);
        for (auto e : qr[i].results) {
            distance = glm_vec3_distance(p[i], e.pos);
            if (!min_distance || distance < min_distance) {
                min_distance = distance;
                enemy = flecs::entity(it.world(), e.e);
            }
        }
        if (min_distance < TurretRange) {
            target[i].target = enemy;
            target[i].distance = min_distance;
        }
    }
}

// Aim turret at target
void AimTarget(flecs::iter& it, Turret* turret, Target* target, Position* p) {
    for (auto i : it) {
        flecs::entity enemy = target[i].target;
        if (enemy && enemy.is_alive()) {
            const Position *tp = enemy.get<Position>();
            if (!tp) {
                continue;
            }

            Position target_p = *tp;
            vec3 diff;
            glm_vec3_sub(target_p, target[i].prev_position, diff);

            target[i].prev_position[0] = target_p.x;
            target[i].prev_position[1] = target_p.y;
            target[i].prev_position[2] = target_p.z;
            float distance = glm_vec3_distance(p[i], target_p);

            // Crude correction for enemy movement and bullet travel time
            if (!turret[i].beam) {
                glm_vec3_scale(diff, distance * 5, diff);
                glm_vec3_add(target_p, diff, target_p);
            }
            target[i].aim_position[0] = target_p.x;
            target[i].aim_position[1] = target_p.y;
            target[i].aim_position[2] = target_p.z;            

            float angle = look_at(p[i], target_p);
            Rotation r = turret[i].head.get<Rotation>()[0];
            r.y = rotate_to(r.y, angle, TurretRotateSpeed * it.delta_time());
            turret[i].head.set<Rotation>(r);
            target[i].angle = angle;
            target[i].lock = (r.y == angle) * (distance < TurretRange);
        }
    }
}

// Countdown until next fire
void FireCountdown(flecs::iter& it, Turret *turret, Target *target) {
    for (auto i : it) {
        turret[i].t_since_fire += it.delta_time();
        if (turret[i].beam_countdown > 0) {
            turret[i].beam_countdown -= it.delta_time();
            if (turret[i].beam_countdown <= 0 || !target[i].lock) {
                turret[i].beam.disable();
            }
        }
    }
}

void AimBeam(flecs::iter& it, Position *p, Turret *turret, Target *target) {
    for (auto i : it) {
        flecs::entity beam = turret[i].beam;
        if (target[i].lock && beam && beam.enabled()) {
            flecs::entity enemy = target[i].target;
            if (!enemy.is_alive()) {
                continue;
            }

            Position pos = p[i];
            const Position *tp = enemy.get<Position>();
            if (!tp) {
                continue;
            }

            Position target_pos = *tp;
            float distance = glm_vec3_distance(p[i], target_pos);

            beam.set<Position>({
                pos.x = (distance / 2),
                pos.y = 0.0,
                pos.z = 0.0
            });

            beam.set<Box>({0.04, 0.04, distance});

            Health *h = enemy.get_mut<Health>();
            h->value -= BeamDamage;

            turret[i].t_since_fire = 0;
        }
    }
}

// Fire bullets at target, alternate firing between cannons
void FireAtTarget(flecs::iter& it, Turret* turret, Target* target, Position* p){
    auto ecs = it.world();
    auto g = it.column<const Game>(4);
    bool is_railgun = it.entity(0).has<Railgun>();

    for (auto i : it) {
        if (turret[i].t_since_fire < turret[i].fire_interval) {
            continue;
        }

        if (target[i].lock) {
            Position pos = p[i];
            float angle = target[i].angle;
            vec3 v, target_p;
            target_p[0] = target[i].aim_position[0];
            target_p[1] = target[i].aim_position[1];
            target_p[2] = target[i].aim_position[2];
            glm_vec3_sub(p[i], target_p, v);
            glm_vec3_normalize(v);

            if (!is_railgun) {
                pos.x += 1.8 * TurretCannonLength * -v[0];
                pos.z += 1.8 * TurretCannonLength * -v[2];
                glm_vec3_scale(v, BulletSpeed * it.delta_time(), v);
                pos.x += sin(angle) * TurretCannonOffset * turret[i].lr;
                pos.y = 0.6;
                pos.z += cos(angle) * TurretCannonOffset * turret[i].lr;
                turret[i].lr = -turret[i].lr;

                ecs.entity().add_instanceof(g->bullet_prefab)
                    .set<Position>(pos)
                    .set<Velocity>({-v[0], 0, -v[2]});
                ecs.entity().add_instanceof(g->fireball_prefab)
                    .set<Position>(pos)
                    .set<Rotation>({0, angle, 0});  
                ecs.entity().add_instanceof(g->shell_prefab)
                    .set<Position>(pos);   
            } else {
                turret[i].beam.enable();
                turret[i].beam_countdown = 2.0;
                pos.x += 0.6 * -v[0];
                pos.y = 0.6;
                pos.z += 0.6 * -v[2];

                ecs.entity().add_instanceof(g->bolt_prefab)
                    .set<Position>(pos)
                    .set<Rotation>({0, angle, 0}); 
            }

            turret[i].t_since_fire = 0;                      
        }
    }
}

// Progress particles
void ProgressParticle(flecs::iter& it, ParticleLifespan *pl) {
    auto p = it.column<const Particle>(2);
    auto box = it.column<Box>(3);
    auto color = it.column<Color>(4);
    auto vel = it.column<Velocity>(5);

    if (box.is_set()) {
        for (auto i : it) {
            box[i].width *= p->size_decay;
            box[i].height *= p->size_decay;
            box[i].depth *= p->size_decay;
        }
    }
    if (color.is_set()) {
        for (auto i : it) {
            color[i].value.r *= p->color_decay;
            color[i].value.g *= p->color_decay;
            color[i].value.b *= p->color_decay;
        }        
    }
    if (vel.is_set()) {
        for (auto i : it) {
            vel[i].x *= p->velocity_decay;
            vel[i].y *= p->velocity_decay;
            vel[i].z *= p->velocity_decay;
        }          
    }
    for (auto i : it) {
        pl[i].t += it.delta_time();
        if (pl[i].t > p->lifespan) {
            it.entity(i).destruct();
        }
    }
}

static
void add_explosion(flecs::world& ecs, const Game& g, Position& p) {
    for (int s = 0; s < 25; s ++) {
        float red = randf(0.4) + 0.6;
        float green = randf(0.3);
        float size = SmokeSize * randf(1.0);
        float radius = 0.5;

        if (green > red) {
            green = red;
        }

        ecs.entity()
            .add_instanceof(g.smoke_prefab)
            .set<Position>({
                p.x + randf(radius) - radius / 2, 
                p.y + randf(radius) - radius / 2,  
                p.z + randf(radius) - radius / 2})
            .set<Box>({size, size, size})
            .set<Color>({red, green, 0});
    }
    for (int s = 0; s < 15; s ++) {
        float x_r = randf(ECS_PI_2);
        float y_r = randf(ECS_PI_2);
        float z_r = randf(ECS_PI_2);
        float speed = randf(5) + 10;
        ecs.entity().add_instanceof(g.spark_prefab)
            .set<Position>({p.x, p.y, p.z}) 
            .set<Velocity>({
                cos(x_r) / speed, cos(y_r) / speed, cos(z_r) / speed});
    }
}

static
void shake_camera(
    const graphics::Camera& cam, CameraController& ctrl, Position& p) 
{
    float d = glm_vec3_distance((float*)cam.position, p);
    if (d < 8.0) {
        ctrl.shake = 0.5 / d;
    }
}

// Check if enemy has been hit by a bullet
void HitTarget(flecs::iter& it, Position* p, Health* h) {
    auto ecs = it.world();
    auto b = it.column<const Box>(3);
    auto q = it.column<const SpatialQuery>(4);
    auto qr = it.column<SpatialQueryResult>(5);

    for (int i : it) {
        float range;
        if (it.is_owned(3)) {
            range = b[i].width / 2;
        } else {
            range = b->width / 2;
        }

        q->query.findn(p[i], range, qr[i].results);
        for (auto e : qr[i].results) {
            auto bullet = ecs.entity(e.e);
            bullet.destruct();
            h[i].value -= BulletDamage;
        }
    }
}

void DestroyEnemy(flecs::iter& it, Health *h, Position *p) {
    auto ecs = it.world();
    auto g = it.column<const Game>(3);
    auto cam = it.column<const graphics::Camera>(4);
    auto cam_ctrl = it.column<CameraController>(5);

    for (auto i : it) {
        if (h[i].value <= 0) {
            it.entity(i).destruct();
            add_explosion(ecs, g[0], p[i]);
            shake_camera(cam[0], cam_ctrl[0], p[i]);
        }
    }
}

// Update enemy color based on health
void UpdateEnemyColor(flecs::iter& it, Color* c, Health* h) {
    for (auto i : it) {
        c[i].value.r = h[i].value;
        c[i].value.g = 0.1 + h[i].value / 5;
        c[i].value.b = 0.1 + h[i].value / 5;
    }
}

// Init Game
flecs::entity init_game(flecs::world& ecs) {
    auto game = ecs.entity("Game");
    Game *g = game.get_mut<Game>();
    g->center = (Position){ to_x(TileCountX / 2), 0, to_z(TileCountZ / 2) };
    g->size = TileCountX * (TileSize + TileSpacing) + 2;
    return game;
}

// Init UI
void init_ui(flecs::world& ecs, flecs::entity game) {
    graphics::Camera camera_data;
    camera_data.set_position(0, CameraHeight, 0);
    camera_data.set_lookat(0, 0, to_z(TileCountZ / 2));
    auto camera = ecs.entity("Camera")
        .set<graphics::Camera>(camera_data)
        .set<CameraController>({-GLM_PI / 2, 0});

    graphics::DirectionalLight light_data;
    light_data.set_position(0, 0, 0);
    light_data.set_direction(0.6, 1.0, 0.4);
    light_data.set_color(0.4, 0.4, 0.37);
    auto light = ecs.entity("Sun")
        .set<graphics::DirectionalLight>(light_data);

    gui::Window window_data;
    window_data.width = 1024;
    window_data.height = 800;
    window_data.title = "Flecs Tower Defense";
    auto window = ecs.entity().set<gui::Window>(window_data);

    gui::Canvas canvas_data;
    canvas_data.background_color = {0.05, 0.07, 0.1};
    canvas_data.ambient_light = {0.3, 0.3, 0.43};
    canvas_data.camera = camera.id();
    canvas_data.directional_light = light.id();
    window.set<gui::Canvas>(canvas_data);

    game.patch<Game>([window](Game& g) {
        g.window = window;
    });
}

// Init level
void init_level(flecs::world& ecs, flecs::entity game) {
    Game *g = game.get_mut<Game>();

    grid<bool> *path = new grid<bool>(TileCountX, TileCountZ);
    path->set(0, 1, true); path->set(1, 1, true); path->set(2, 1, true);
    path->set(3, 1, true); path->set(4, 1, true); path->set(5, 1, true);
    path->set(6, 1, true); path->set(7, 1, true); path->set(8, 1, true);
    path->set(8, 2, true); path->set(8, 3, true); path->set(7, 3, true);
    path->set(6, 3, true); path->set(5, 3, true); path->set(4, 3, true);
    path->set(3, 3, true); path->set(2, 3, true); path->set(1, 3, true);
    path->set(1, 4, true); path->set(1, 5, true); path->set(1, 6, true);
    path->set(1, 7, true); path->set(1, 8, true); path->set(2, 8, true);
    path->set(3, 8, true); path->set(4, 8, true); path->set(4, 7, true);
    path->set(4, 6, true); path->set(4, 5, true); path->set(5, 5, true);
    path->set(6, 5, true); path->set(7, 5, true); path->set(8, 5, true);
    path->set(8, 6, true); path->set(8, 7, true); path->set(7, 7, true);
    path->set(6, 7, true); path->set(6, 8, true); path->set(6, 9, true);
    path->set(7, 9, true); path->set(8, 9, true); path->set(9, 9, true);
    
    transform::Position2 spawn_point = {
        to_x(TileCountX - 1), 
        to_z(TileCountZ - 1)
    };

    g->level = ecs.entity()
        .set<Level>({path, spawn_point});

    for (int x = 0; x < TileCountX; x ++) {
        for (int z = 0; z < TileCountZ; z++) {
            float xc = to_x(x);
            float zc = to_z(z);

            auto t = ecs.entity();
            if (path[0](x, z)) {
                t.add_instanceof(g->path_prefab);
            } else {
                t.add_instanceof(g->tile_prefab);

                if (randf(1) > 0.3) {
                    ecs.entity().add_instanceof(g->cannon_prefab)
                        .set<Position>({xc, TileHeight / 2, zc});
                } else {
                    ecs.entity().add_instanceof(g->railgun_prefab)
                        .set<Position>({xc, TileHeight / 2, zc});
                }
            }

            t.set<Position>({xc, 0, zc});                
        }
    }                                               
}

// Init prefabs
void init_prefabs(flecs::world& ecs, flecs::entity game) {
    Game *g = game.get_mut<Game>();

    g->tile_prefab = ecs.prefab()
        .set<Color>({0.2, 0.2, 0.2})
        .set<Box>({TileSize, TileHeight, TileSize});

    g->path_prefab = ecs.prefab()
        .set<Color>({0.3, 0.3, 0.3})
        .set<Specular>({0.25, 20})
        .set<Box>({TileSize + TileSpacing, PathHeight, TileSize + TileSpacing});

    auto bullet_query_trait = ecs.type()
        .add_trait<SpatialQueryResult, Bullet>();

    g->enemy_prefab = ecs.prefab()
        .add<Enemy>()
        .add<Health>()
        .set<Color>({1.0, 0.3, 0.3})
        .set<Box>({EnemySize, EnemySize, EnemySize})
        .set<Specular>({2.0, 150})
        .set_trait<SpatialQuery, Bullet>({
            flecs::squery(ecs, "ANY:Bullet", g->center, g->size)
        })
        .add_trait<SpatialQueryResult, Bullet>()
        .add_owned<Color>()
        .add_owned<Health>()
        .add_owned(bullet_query_trait);

    g->bullet_prefab = ecs.prefab("BulletPrefab")
        .add<Bullet>()
        .set<Color>({0, 0, 0})
        .set<Box>({BulletSize, BulletSize, BulletSize})
        .set<Particle>({
            1.0, 1.0, 1.0, BulletLifespan
        })
        .add_owned<ParticleLifespan>();

    g->beam_prefab = ecs.prefab("RailBulletPrefab")
        .add<Bullet>()
        .set<Color>({0.3, 0.6, 1})
        .set<Emissive>({2})
        .set<Box>({BulletSize, BulletSize, BulletSize})
        .add_owned<Box>();

    g->shell_prefab = ecs.prefab("ShellPrefab")
        .set<Color>({0, 0, 0})
        .set<Box>({ShellSize, ShellSize, ShellSize})
        .set<Particle>({
            1.0, 1.0, 1.0, ShellLifespan
        })
        .set<Velocity>({0, -ShellSpeed, 0.0})
        .add_owned<Box>()
        .add_owned<Color>()
        .add_owned<ParticleLifespan>()
        .add_owned<Velocity>();

    g->fireball_prefab = ecs.prefab("FireballPrefab")
        .set<Color>({1.0, 1.0, 0.5})
        .set<Emissive>({1.6})
        .set<Box>({FireballSize, FireballSize, FireballSize})
        .set<Particle>({
            FireballSizeDecay, 1.0, 1.0, FireballLifespan
        })
        .add_owned<Color>()
        .add_owned<Box>()
        .add_owned<ParticleLifespan>();

    g->bolt_prefab = ecs.prefab("BoltPrefab")
        .set<Color>({0.3, 0.6, 1.0})
        .set<Emissive>({3})
        .set<Box>({BoltSize, BoltSize, BoltSize})
        .set<Particle>({
            BoltSizeDecay, 1.0, 1.0, BoltLifespan
        })
        .add_owned<Color>()
        .add_owned<Box>()
        .add_owned<ParticleLifespan>();

    g->smoke_prefab = ecs.prefab("SmokePrefab")
        .set<Color>({0, 0, 0})
        .set<Emissive>({8.0})
        .set<Box>({SmokeSize, SmokeSize, SmokeSize})
        .set<Particle>({
            SmokeSizeDecay, SmokeColorDecay, 1.0, SmokeLifespan
        })
        .set<Velocity>({0, 0.003, 0})
        .add_owned<Color>()
        .add_owned<Box>()
        .add_owned<Velocity>()
        .add_owned<ParticleLifespan>();

    g->spark_prefab = ecs.prefab("SparkPrefab")
        .set<Color>({1.0, 0.5, 0.5})
        .set<Emissive>({2.0})
        .set<Box>({SparkSize, SparkSize, SparkSize})
        .set<Particle>({
            SparkSizeDecay, 1.0, SparkVelocityDecay, SparkLifespan
        })
        .add_owned<Color>()
        .add_owned<Box>()
        .add_owned<ParticleLifespan>();         

    auto enemy_query_trait = ecs.type()
        .add_trait<SpatialQueryResult, Enemy>();

    auto turret_prefab = ecs.prefab()
        .set_trait<SpatialQuery, Enemy>({
            flecs::squery(ecs, "ANY:Enemy", g->center, g->size)
        })
        .add_trait<SpatialQueryResult, Enemy>()
        .add_owned<Target>()
        .add_owned(enemy_query_trait);

    g->cannon_prefab = ecs.prefab()
        .add_instanceof(turret_prefab)
        .add<Cannon>()
        .set<Turret>({0.05})
        .add_owned<Turret>();

        ecs.prefab()
            .add_childof(g->cannon_prefab)
            .set<Color>({0.1, 0.1, 0.1})
            .set<Box>({0.3, 0.1, 0.3})
            .set<Position>({0, 0.05, 0});

        ecs.prefab()
            .add_childof(g->cannon_prefab)
            .set<Color>({0.15, 0.15, 0.15})
            .set<Box>({0.2, 0.3, 0.2})
            .set<Position>({0, 0.15, 0});

        auto cannon_head_material = ecs.prefab("CannonHeadMaterial")
            .set<Color>({0.35, 0.4, 0.3})
            .set<Specular>({0.5, 25});

        auto cannon_head = ecs.prefab("TurretHead")
            .add_childof(g->cannon_prefab)
            .add_instanceof(cannon_head_material)
            .set<Box>({0.4, 0.2, 0.4})
            .set<Position>({0, 0.4, 0})
            .set<Rotation>({0, 0.0, 0});

            auto cannon = ecs.prefab("Cannon")
                .set<Color>({0.1, 0.1, 0.1})
                .set<Box>({0.4, 0.07, 0.07})
                .set<Specular>({2, 75});

            ecs.prefab("CannonLeft").add_instanceof(cannon)
                .add_childof(cannon_head)
                .set<Position>({TurretCannonLength, 0.0, -TurretCannonOffset}); 

            ecs.prefab("CannonRight").add_instanceof(cannon)
                .add_childof(cannon_head)
                .set<Position>({TurretCannonLength, 0.0, TurretCannonOffset});                         

    g->railgun_prefab = ecs.prefab()
        .add_instanceof(turret_prefab)
        .add<Railgun>()
        .set<Turret>({1.0})
        .add_owned<Turret>();        

        ecs.prefab()
            .add_childof(g->railgun_prefab)
            .set<Color>({0.1, 0.1, 0.1})
            .set<Box>({0.3, 0.3, 0.3})
            .set<Position>({0, 0.15, 0});

        auto railgun_head_material = ecs.prefab("RailgunHeadMaterial")
            .set<Color>({0.1, 0.1, 0.1})
            .set<Specular>({1.5, 128});

        auto railgun_glow_material = ecs.prefab("RailgunGlowMaterial")
            .set<Color>({0.1, 0.3, 1.0})
            .set<Emissive>({3.8});

        auto railgun_head = ecs.prefab("TurretHead")
            .add_childof(g->railgun_prefab)
            .set<Position>({0.0, 0.4, 0})
            .set<Rotation>({0, 0.0, 0});

            ecs.prefab()
                .add_childof(railgun_head)
                .add_instanceof(railgun_head_material)
                .set<Box>({0.45, 0.15, 0.08})
                .set<Position>({0.05, 0.0, -0.15});
                
            ecs.prefab()
                .add_childof(railgun_head)
                .add_instanceof(railgun_head_material)
                .set<Box>({0.45, 0.15, 0.08})
                .set<Position>({0.05, 0.0, 0.15});                

            ecs.prefab()
                .add_childof(railgun_head)
                .add_instanceof(railgun_glow_material)
                .set<Box>({0.4, 0.1, 0.05})
                .set<Position>({0.05, 0.0, -0.10});

            ecs.prefab()
                .add_childof(railgun_head)
                .add_instanceof(railgun_glow_material)
                .set<Box>({0.4, 0.1, 0.05})
                .set<Position>({0.05, 0.0, 0.10});

            ecs.prefab()
                .add_childof(railgun_head)
                .add_instanceof(railgun_head_material)
                .set<Box>({0.8, 0.25, 0.15})
                .set<Position>({0.12, 0, 0.0});  

            ecs.prefab("TurretBeam").add_instanceof(g->beam_prefab)
                .add_childof(railgun_head)
                .set<Position>({1, 0, 0})
                .set<Box>({0.04, 0.04, 2.0})
                .set<Rotation>({0,(float)GLM_PI / 2.0f, 0});

    // When Turret is set, initialize it with the head child
    ecs.system<Turret>()
        .kind(flecs::OnSet)
        .each([](flecs::entity e, Turret& t) {
            t.head = e.lookup("TurretHead");
            t.beam = t.head.lookup("TurretBeam");
            if (t.beam) {
                t.beam.disable();
            }
        });
}

// Init systems
void init_systems(flecs::world& ecs) {
    // Move camera with keyboard
    ecs.system<CameraController>(
        "MoveCamera", "$:Input, Camera:flecs.components.graphics.Camera")
        .iter(MoveCamera);

    // Spawn enemies periodically
    ecs.system<>(
        "SpawnEnemy", "Game:Game")
        .interval(EnemySpawnInterval)
        .iter(SpawnEnemy);

    // Move enemies
    ecs.system<Position, Direction>(
        "MoveEnemy", "Game:Game, ANY:Enemy")
        .iter(MoveEnemy);

    // Clear invalid target for turrets
    ecs.system<Target, Position>(
        "ClearTarget")
        .iter(ClearTarget);

    // Find target for turrets
    ecs.system<Target, Position>(
        "FindTarget", "SHARED:SpatialQuery FOR Enemy, SpatialQueryResult FOR Enemy")
        .iter(FindTarget);

    // Aim turret at enemies
    ecs.system<Turret, Target, Position>(
        "AimTarget")
        .iter(AimTarget);

    // Countdown until next fire
    ecs.system<Turret, Target>(
        "FireCountdown")
        .iter(FireCountdown);

    // Aim beam at target
    ecs.system<Position, Turret, Target>(
        "AimBeam")
        .iter(AimBeam);

    // Fire bullets at enemies
    ecs.system<Turret, Target, Position>(
        "FireAtTarget", "Game:Game")
        .interval(TurretFireInterval)
        .iter(FireAtTarget);

    ecs.system<ParticleLifespan>(
        "ProgressParticle", "SHARED:Particle, ?Box, ?Color, ?Velocity")
        .iter(ProgressParticle);

    // Test for collisions with enemies
    ecs.system<Position, Health>(
        "HitTarget", 
        "ANY:Box, SHARED:SpatialQuery FOR Bullet,SpatialQueryResult FOR Bullet")
        .iter(HitTarget);

    ecs.system<Health, Position>(
        "DestroyEnemy", 
        "Game:Game, Camera:flecs.components.graphics.Camera, Camera:CameraController, ANY:Enemy")
        .iter(DestroyEnemy);

    ecs.system<Color, Health>(
        "UpdateEnemyColor")
        .iter(UpdateEnemyColor);
}

int main(int argc, char *argv[]) {
    flecs::world ecs;
    ecs.import<flecs::components::transform>();
    ecs.import<flecs::components::graphics>();
    ecs.import<flecs::components::geometry>();
    ecs.import<flecs::components::gui>();
    ecs.import<flecs::components::physics>();
    ecs.import<flecs::components::input>();
    ecs.import<flecs::systems::transform>();
    ecs.import<flecs::systems::physics>();
    ecs.import<flecs::systems::sdl2>();
    ecs.import<flecs::systems::sokol>();

    // Add aliases for components from modules for more readable queries
    ecs.use<SpatialQuery>();
    ecs.use<SpatialQueryResult>();
    ecs.use<Input>();
    ecs.use<Position>("Position");
    ecs.use<Rotation>("Rotation");
    ecs.use<Velocity>("Velocity");
    ecs.use<Color>();
    ecs.use<Box>();

    auto game = init_game(ecs);
    init_ui(ecs, game);
    init_prefabs(ecs, game);
    init_level(ecs, game);
    init_systems(ecs);

    ecs.set_target_fps(60);

    int i = 0;
    ecs_time_t t;
    ecs_time_measure(&t);
    while(ecs.progress()) { 
        i ++;
        if (!(i % 60)) {
            printf("t = %f\n", ecs_time_measure(&t) / 60);
        }
    }
}
