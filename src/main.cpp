#include <iostream>
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
using Color = graphics::Rgb;
using Specular = graphics::Specular;
using Emissive = graphics::Emissive;
using Box = geometry::Box;

#define ECS_PI_2 ((float)(GLM_PI * 2))

// Game constants
static const float EnemySize = 0.7;
static const float EnemySpeed = 4.0;
static const float EnemySpawnInterval = 0.2;

static const float RecoilAmount = 0.3;
static const float DecreaseRecoilRate = 1.5;
static const float HitCooldownRate = 1.0;
static const float HitCooldownInitialValue = 0.25;

static const float TurretRotateSpeed = 4.0;
static const float TurretFireInterval = 0.12;
static const float TurretRange = 5.0;
static const float TurretCannonOffset = 0.2;
static const float TurretCannonLength = 0.6;

static const float BulletSize = 0.1;
static const float BulletSpeed = 24.0;
static const float BulletLifespan = 0.5;
static const float BulletDamage = 0.02;

static const float IonSize = 0.05;
static const float IonLifespan = 1.0;
static const float IonDecay = 0.1;

static const float BeamFireInterval = 0.1;
static const float BeamDamage = 0.275;
static const float BeamSize = 0.04;

static const float NozzleFlashSize = 0.2;
static const float NozzleFlashSizeDecay = 0.001;
static const float NozzleFlashLifespan = 0.1;

static const float BoltSize = 0.2;
static const float BoltSizeDecay = 0.001;
static const float BoltLifespan = 1.0;

static const float SmokeSize = 1.5;
static const float ExplodeRadius = 1.0;
static const float SmokeSizeDecay = 0.4;
static const float SmokeColorDecay = 0.0001;
static const float SmokeLifespan = 4.0;
static const int SmokeParticleCount = 50;

static const float SparkSize = 0.15;
static const float SparkLifespan = 0.4;
static const float SparkSizeDecay = 0.025;
static const float SparkVelocityDecay = 0.025;
static const float SparkInitialVelocity = 9.0;
static const int SparkParticleCount = 20;

static const float TileSize = 3.0;
static const float TileHeight = 0.5;
static const float PathHeight = 0.1;
static const float TileSpacing = 0.00;
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

// Components

struct Game {
    flecs::entity window;
    flecs::entity level;
    
    Position center;
    float size;        
};

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

struct Enemy { };

struct Direction {
    int value;
};

struct Health {
    Health() {
        value = 1.0;
    }
    float value;
};

struct Bullet { };

struct Turret { 
    Turret(float fire_interval_arg = 1.0) {
        lr = 1;
        t_since_fire = 0;
        fire_interval = fire_interval_arg;
    }

    float fire_interval;
    float t_since_fire;
    int lr;
};

struct Recoil {
    float value;
};

struct HitCooldown {
    float value;
};

struct Laser { };

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

// Prefab types
namespace prefabs {
    struct Tree {
        struct Trunk { };
        struct Canopy { };
    };

    struct Tile { };
    struct Path { };
    struct Enemy { };

    struct materials {
        struct Beam { };
        struct Metal { };
        struct CannonHead { };
        struct LaserLight { };
    };

    struct Particle { };
    struct Bullet { };
    struct NozzleFlash { };
    struct Smoke { };
    struct Spark { };
    struct Ion { };
    struct Bolt { };

    struct Turret {
        struct Base { };
        struct Head { };
    };

    struct Cannon {
        struct Head {
            struct BarrelLeft { };
            struct BarrelRight { };
        };
        struct Barrel { };
    };

    struct Laser {
        struct Head {
            struct Beam { };
        };
    };
}

// Scope for systems
struct systems { };

// Scope for level entities (tile, path)
struct level { };

// Scope for turrets
struct turrets { };

// Scope for enemies
struct enemies { };

// Scope for particles
struct particles { };

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

void SpawnEnemy(flecs::iter& it, const Game *g) {
    const Level* lvl = g->level.get<Level>();

    it.world().entity().child_of<enemies>().is_a<prefabs::Enemy>()
        .set<Direction>({0})
        .set<Position>({
            lvl->spawn_point.x, -1.2, lvl->spawn_point.y
        });
}

void MoveEnemy(flecs::iter& it, size_t i,
    Position& p, Direction& d, const Game& g)
{
    const Level* lvl = g.level.get<Level>();

    if (find_path(p, d, lvl)) {
        it.entity(i).destruct(); // Enemy made it to the end
    } else {
        p.x += dir[d.value].x * EnemySpeed * it.delta_time();
        p.z += dir[d.value].y * EnemySpeed * it.delta_time();
    }
}

void ClearTarget(Target& target, Position& p) {
    flecs::entity t = target.target;
    if (t) {
        if (!t.is_alive()) {
            // Target was destroyed or made it to the end
            target.target = flecs::entity::null();
            target.lock = false;
        } else {
            Position target_pos = t.get<Position>()[0];
            float distance = glm_vec3_distance(p, target_pos);
            if (distance > TurretRange) {
                // Target is out of range
                target.target = flecs::entity::null();
                target.lock = false;
            }
        }
    }
}

void FindTarget(flecs::iter& it, size_t i,
    Turret& turret, Target& target, Position& p) 
{
    auto q = it.field<const SpatialQuery>(4);
    auto qr = it.field<SpatialQueryResult>(5);

    if (target.target) {
        // Already has a target
        return;
    }

    flecs::entity enemy;
    float distance = 0, min_distance = 0;

    // Find all enemies around the turret's position within TurretRange
    q->findn(p, TurretRange, qr[i]);
    for (auto e : qr[i]) {
        distance = glm_vec3_distance(p, e.pos);
        if (distance > TurretRange) {
            continue;
        }

        if (!min_distance || distance < min_distance) {
            min_distance = distance;
            enemy = it.world().entity(e.id);
        }
    }

    if (min_distance) {
        // Select the closest enemy in range as target
        target.target = enemy;
        target.distance = min_distance;
    }
}

void AimTarget(flecs::iter& it, size_t i,
    Turret& turret, Target& target, Position& p) 
{
    flecs::entity enemy = target.target;
    if (enemy && enemy.is_alive()) {
        flecs::entity e = it.entity(i);

        Position target_p = enemy.get<Position>()[0];
        vec3 diff;
        glm_vec3_sub(target_p, target.prev_position, diff);

        target.prev_position[0] = target_p.x;
        target.prev_position[1] = target_p.y;
        target.prev_position[2] = target_p.z;
        float distance = glm_vec3_distance(p, target_p);

        // Crude correction for enemy movement and bullet travel time
        flecs::entity beam = e.target<prefabs::Laser::Head::Beam>();
        if (!beam) {
            glm_vec3_scale(diff, distance * 5, diff);
            glm_vec3_add(target_p, diff, target_p);
        }

        target.aim_position[0] = target_p.x;
        target.aim_position[1] = target_p.y;
        target.aim_position[2] = target_p.z;            

        float angle = look_at(p, target_p);

        flecs::entity head = e.target<prefabs::Turret::Head>();
        Rotation r = head.get<Rotation>()[0];
        r.y = rotate_to(r.y, angle, TurretRotateSpeed * it.delta_time());
        head.set<Rotation>(r);
        target.angle = angle;
        target.lock = (r.y == angle) * (distance < TurretRange);
    }
}

void FireCountdown(flecs::iter& it, size_t i,
    Turret& turret, Target& target) 
{
    turret.t_since_fire += it.delta_time();
}

void FireAtTarget(flecs::iter& it, size_t i,
    Turret& turret, Target& target, Position& p)
{
    auto ecs = it.world();
    bool is_laser = it.is_set(4);
    flecs::entity e = it.entity(i);

    if (turret.t_since_fire < turret.fire_interval) {
        // Cooldown so we don't shoot too fast
        return;
    }

    if (target.target && target.lock) {
        Position pos = p;
        float angle = target.angle;
        vec3 v, target_p;
        target_p[0] = target.aim_position[0];
        target_p[1] = target.aim_position[1];
        target_p[2] = target.aim_position[2];
        glm_vec3_sub(p, target_p, v);
        glm_vec3_normalize(v);

        if (!is_laser) {
            // Regular cannon with two barrels
            pos.x += 1.7 * TurretCannonLength * -v[0];
            pos.z += 1.7 * TurretCannonLength * -v[2];
            glm_vec3_scale(v, BulletSpeed, v);
            pos.x += sin(angle) * TurretCannonOffset * turret.lr;
            pos.y = -1.1;
            pos.z += cos(angle) * TurretCannonOffset * turret.lr;

            // Alternate between left and right barrel
            flecs::entity barrel;
            if (turret.lr == -1) {
                barrel = e.target<prefabs::Cannon::Head::BarrelLeft>();
            } else {
                barrel = e.target<prefabs::Cannon::Head::BarrelRight>();
            }
            turret.lr = -turret.lr;

            // Move active barrel backwards to simulate recoil
            barrel.set<Recoil>({ RecoilAmount });

            // Create a bullet and nozzle flash
            ecs.entity().is_a<prefabs::Bullet>()
                .set<Position>(pos)
                .set<Velocity>({-v[0], 0, -v[2]});
            ecs.entity().is_a<prefabs::NozzleFlash>()
                .set<Position>(pos)
                .set<Rotation>({0, angle, 0});
        } else {
            // Enable laser beam
            e.target<prefabs::Laser::Head::Beam>().enable();
            pos.x += 1.4 * -v[0];
            pos.y = -1.1;
            pos.z += 1.4 * -v[2];
            ecs.scope<particles>().entity().is_a<prefabs::Bolt>()
                .set<Position>(pos)
                .set<Rotation>({0, angle, 0}); 
        }

        turret.t_since_fire = 0;
    }
}

void BeamControl(flecs::iter& it, size_t i,
    Position& p, Turret& turret, Target& target) 
{
    flecs::entity beam = it.entity(i).target<prefabs::Laser::Head::Beam>();
    if (beam && (!target.target || !target.lock)) {
        // Disable beam if laser turret has no target
        beam.disable();
        beam.set<Box>({0.0, 0.0, 0});
        return;
    }

    if (target.lock && beam && beam.enabled()) {
        flecs::entity enemy = target.target;
        if (!enemy.is_alive()) {
            return;
        }

        // Position beam at enemy
        Position pos = p;
        Position target_pos = enemy.get<Position>()[0];
        float distance = glm_vec3_distance(p, target_pos);
        beam.set<Position>({ (distance / 2), -0.1, 0.0 });
        beam.set<Box>({BeamSize, BeamSize, distance});

        // Subtract health from enemy as long as beam is firing
        enemy.get([&](Health& h, HitCooldown& hc) {
            h.value -= BeamDamage * it.delta_time();
            hc.value = HitCooldownInitialValue;
        });

        // Create ion trail particles
        if (randf(1.0) > 0.5) {
            vec3 v;
            glm_vec3_sub(pos, target_pos, v);
            glm_vec3_normalize(v);

            float ion_d = randf(distance - 0.7) + 0.7;
            Position ion_pos = {pos.x - ion_d * v[0], -1.1, pos.z - ion_d * v[2]};
            Velocity ion_v = {
                randf(0.02),
                randf(0.02) + 0.01f,
                randf(0.02)
            };

            it.world().scope<particles>().entity().is_a<prefabs::Ion>()
                .set<Position>(ion_pos)
                .set<Velocity>(ion_v);
        }
    }
}

void ApplyRecoil(Position& p, const Recoil& r) {
   p.x = TurretCannonLength - r.value; 
}

void DecreaseRecoil(flecs::iter& it, size_t, Recoil& r) {
   r.value -= it.delta_time() * DecreaseRecoilRate;
   if (r.value < 0) {
    r.value = 0;
   }
}

void DecreaseHitCoolDown(flecs::iter& it, size_t, HitCooldown& hc) {
   hc.value -= it.delta_time() * HitCooldownRate;
   if (hc.value < 0) {
    hc.value = 0;
   }
}

void ProgressParticle(flecs::iter& it,
    ParticleLifespan *pl, const Particle *p, Box *box, Color *color, Velocity *vel)
{
    if (it.is_set(3)) { // Size
        for (auto i : it) {
            box[i].width *= pow(p->size_decay, it.delta_time());
            box[i].height *= pow(p->size_decay, it.delta_time());
            box[i].depth *= pow(p->size_decay, it.delta_time());
        }
    }
    if (it.is_set(4)) { // Color
        for (auto i : it) {
            color[i].r *= pow(p->color_decay, it.delta_time());
            color[i].g *= pow(p->color_decay, it.delta_time());
            color[i].b *= pow(p->color_decay, it.delta_time());
        }        
    }
    if (it.is_set(5)) { // Velocity
        for (auto i : it) {
            vel[i].x *= pow(p->velocity_decay, it.delta_time());
            vel[i].y *= pow(p->velocity_decay, it.delta_time());
            vel[i].z *= pow(p->velocity_decay, it.delta_time());
        }          
    }
    for (auto i : it) { // Lifespan
        pl[i].t += it.delta_time();
        if (pl[i].t > p->lifespan) {
            it.entity(i).destruct();
        }
    }
}

void HitTarget(flecs::iter& it, size_t i,
    Position& p, Health& h, Box& b, HitCooldown& hit_cooldown)
{
    auto q = it.field<const SpatialQuery>(5);
    auto qr = it.field<SpatialQueryResult>(6);
    
    float range = b.width / 2;

    // Test whether bullet hit an enemy
    q->findn(p, range, qr[i]);
    for (auto e : qr[i]) {
        it.world().entity(e.id).destruct();
        h.value -= BulletDamage;
        hit_cooldown.value = HitCooldownInitialValue; // For color effect
    }
}

static
void explode(flecs::world& ecs, Position& p) {
    // Create explosion particles that fade into smoke
    for (int s = 0; s < SmokeParticleCount; s ++) {
        float red = randf(0.5) + 0.7;
        float green = randf(0.7);
        float blue = randf(0.4);
        float size = SmokeSize * randf(1.0);
        if (green > red) {
            green = red;
        }
        if (blue > green) {
            blue = green;
        }

        ecs.scope<particles>().entity().is_a<prefabs::Smoke>()
            .set<Position>({
                p.x + randf(ExplodeRadius) - ExplodeRadius / 2, 
                p.y + randf(ExplodeRadius) - ExplodeRadius / 2,  
                p.z + randf(ExplodeRadius) - ExplodeRadius / 2})
            .set<Box>({size, size, size})
            .set<Color>({red, green, blue});
    }

    // Create sparks
    for (int s = 0; s < SparkParticleCount; s ++) {
        float x_r = randf(ECS_PI_2);
        float y_r = randf(ECS_PI_2);
        float z_r = randf(ECS_PI_2);
        float speed = randf(SparkInitialVelocity) + 2.0;
        ecs.scope<particles>().entity().is_a<prefabs::Spark>()
            .set<Position>({p.x, p.y, p.z}) 
            .set<Velocity>({
                cos(x_r) * speed, cos(y_r) * speed, cos(z_r) * speed});
    }
}

void DestroyEnemy(flecs::entity e, Health& h, Position& p) {
    flecs::world ecs = e.world();
    if (h.value <= 0) {
        e.destruct();
        explode(ecs, p);
    }
}

void UpdateEnemyColor(Color& c, const Health& h, const HitCooldown& hc) {
    // Increase brightness after hit
    c.r = 0.05 + hc.value;
    c.g = 0.05 + hc.value * 0.1;

    // Increase brightness as health decreases
    c.r += (0.3 - h.value * 0.3);
    c.g += (0.05 - h.value * 0.05);
}

void init_game(flecs::world& ecs) {
    Game *g = ecs.get_mut<Game>();
    g->center = { to_x(TileCountX / 2), 0, to_z(TileCountZ / 2) };
    g->size = TileCountX * (TileSize + TileSpacing) + 2;
}

void init_ui(flecs::world& ecs) {
    graphics::Camera camera_data = {};
    camera_data.set_up(0, -1, 0);
    camera_data.set_fov(20);
    camera_data.near_ = 1.0;
    camera_data.far_ = 100.0;
    auto camera = ecs.entity("Camera")
        .add(flecs::game::CameraController)
        .set<Position>({0, -8.0, -9.0})
        .set<Rotation>({0.5})
        .set<graphics::Camera>(camera_data);

    graphics::DirectionalLight light_data = {};
    light_data.set_direction(0.3, -1.0, 0.5);
    light_data.set_color(0.98, 0.95, 0.8);
    auto light = ecs.entity("Sun")
        .set<graphics::DirectionalLight>(light_data);

    gui::Canvas canvas_data = {};
    canvas_data.width = 1400;
    canvas_data.height = 1000;
    canvas_data.title = (char*)"Flecs Tower Defense";
    canvas_data.camera = camera.id();
    canvas_data.directional_light = light.id();
    canvas_data.ambient_light = {0.06, 0.05, 0.18};
    canvas_data.background_color = {0.15, 0.4, 0.6};
    canvas_data.fog_density = 0.65;
    ecs.entity().set<gui::Canvas>(canvas_data);
}

// Init level
void init_level(flecs::world& ecs) {
    Game *g = ecs.get_mut<Game>();

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

    g->level = ecs.entity().set<Level>({path, spawn_point});

    ecs.entity()
        .set<Position>({0, 2.5, to_z(TileCountZ / 2 - 0.5)})
        .set<Box>({to_x(TileCountX + 0.5) * 2, 5, to_z(TileCountZ + 2)})
        .set<Color>({0.11, 0.15, 0.1});

    for (int x = 0; x < TileCountX; x ++) {
        for (int z = 0; z < TileCountZ; z++) {
            float xc = to_x(x);
            float zc = to_z(z);

            auto t = ecs.scope<level>().entity().set<Position>({xc, 0, zc});
            if (path[0](x, z)) {
                t.is_a<prefabs::Path>();
            } else {
                t.is_a<prefabs::Tile>();

                auto e = ecs.entity().set<Position>({xc, -TileHeight / 2, zc});
                if (randf(1) > 0.65) {
                    e.child_of<level>();
                    e.is_a<prefabs::Tree>();
                } else {
                    e.child_of<turrets>();
                    if (randf(1) > 0.3) {
                        e.is_a<prefabs::Cannon>();
                    } else {
                        e.is_a<prefabs::Laser>();
                        e.target<prefabs::Laser::Head::Beam>().disable();
                    }
                }
            }           
        }
    }                          
}

void init_prefabs(flecs::world& ecs) {
    Game *g = ecs.get_mut<Game>();

    ecs.prefab<prefabs::Tree>();
        ecs.prefab<prefabs::Tree::Trunk>()
            .set<Position>({0, -0.75, 0})
            .set<Color>({0.25, 0.2, 0.1})
            .set<Box>({0.5, 1.5, 0.5});
        ecs.prefab<prefabs::Tree::Canopy>()
            .set<Position>({0, -2.0, 0})
            .set<Color>({0.2, 0.3, 0.15})
            .set<Box>({1.5, 1.8, 1.5});

    ecs.prefab<prefabs::Tile>()
        .set<Color>({0.2, 0.34, 0.15})
        .set<Specular>({0.25, 20})
        .set<Box>({TileSize, TileHeight, TileSize});

    ecs.prefab<prefabs::Path>()
        .set<Color>({0.2, 0.2, 0.2})
        .set<Specular>({0.5, 50})
        .set<Box>({TileSize + TileSpacing, PathHeight, TileSize + TileSpacing});

    ecs.prefab<prefabs::materials::Beam>()
        .set<Color>({0.1, 0.4, 1})
        .set<Emissive>({10.0});

    ecs.prefab<prefabs::materials::Metal>()
        .set<Color>({0.1, 0.1, 0.1})
        .set<Specular>({1.5, 128});

    ecs.prefab<prefabs::materials::CannonHead>()
        .set<Color>({0.35, 0.4, 0.3})
        .set<Specular>({0.5, 25});

    ecs.prefab<prefabs::materials::LaserLight>()
        .set<Color>({0.1, 0.3, 1.0})
        .set<Emissive>({3.0});

    ecs.prefab<prefabs::Enemy>()
        .is_a<prefabs::materials::Metal>()
        .add<Enemy>()
        .add<Health>().override<Health>()
        .set_override<Color>({0.05, 0.05, 0.05})
        .set<Box>({EnemySize, EnemySize, EnemySize})
        .set<Specular>({4.0, 512})
        .set<SpatialQuery, Bullet>({g->center, g->size})
        .set_override<HitCooldown>({})
        .add<SpatialQueryResult, Bullet>()
        .override<SpatialQueryResult, Bullet>();

    ecs.prefab<prefabs::Particle>()
        .override<ParticleLifespan>()
        .override<Color>()
        .override<Box>();

    ecs.prefab<prefabs::Bullet>().is_a<prefabs::Particle>()
        .add<Bullet>()
        .set<Color>({0, 0, 0})
        .set<Box>({BulletSize, BulletSize, BulletSize})
        .set<Particle>({
            1.0, 1.0, 1.0, BulletLifespan
        });

    ecs.prefab<prefabs::NozzleFlash>().is_a<prefabs::Particle>()
        .set<Color>({1.0, 0.5, 0.3})
        .set<Emissive>({5.0})
        .set<Box>({NozzleFlashSize, NozzleFlashSize, NozzleFlashSize})
        .set<Particle>({
            NozzleFlashSizeDecay, 1.0, 1.0, NozzleFlashLifespan
        });

    ecs.prefab<prefabs::Smoke>().is_a<prefabs::Particle>()
        .set<Color>({0, 0, 0})
        .set<Emissive>({20.0})
        .set<Box>({SmokeSize, SmokeSize, SmokeSize})
        .set<Particle>({
            SmokeSizeDecay, SmokeColorDecay, 1.0, SmokeLifespan
        })
        .set<Velocity>({0, -0.8, 0})
        .override<Velocity>();

    ecs.prefab<prefabs::Spark>().is_a<prefabs::Particle>()
        .set<Color>({1.0, 0.5, 0.2})
        .set<Emissive>({5.0})
        .set<Box>({SparkSize, SparkSize, SparkSize})
        .set<Particle>({
            SparkSizeDecay, 1.0, SparkVelocityDecay, SparkLifespan
        });

    ecs.prefab<prefabs::Ion>()
        .is_a<prefabs::materials::Beam>()
        .is_a<prefabs::Particle>()
        .set<Box>({IonSize, IonSize, IonSize})
        .set<Particle>({
            IonDecay, 1.0, 1.0, IonLifespan
        });

    ecs.prefab<prefabs::Bolt>()
        .is_a<prefabs::materials::Beam>()
        .is_a<prefabs::Particle>()
        .set<Box>({BoltSize, BoltSize, BoltSize})
        .set<Particle>({
            BoltSizeDecay, 1.0, 1.0, BoltLifespan
        });

    ecs.prefab<prefabs::Turret>()
        .set<SpatialQuery, Enemy>({ g->center, g->size })
        .add<SpatialQueryResult, Enemy>()
        .override<SpatialQueryResult, Enemy>()
        .override<Target>()
        .override<Turret>();

        ecs.prefab<prefabs::Turret::Base>().slot()
            .set<Position>({0, 0, 0});

            ecs.prefab().is_a<prefabs::materials::Metal>()
                .child_of<prefabs::Turret::Base>()
                .set<Box>({0.6, 0.2, 0.6})
                .set<Position>({0, -0.1, 0});

            ecs.prefab().is_a<prefabs::materials::Metal>()
                .child_of<prefabs::Turret::Base>()
                .set<Box>({0.4, 0.6, 0.4})
                .set<Position>({0, -0.3, 0});

        ecs.prefab<prefabs::Turret::Head>().slot();

    ecs.prefab<prefabs::Cannon>().is_a<prefabs::Turret>()
        .set<Turret>({TurretFireInterval});

        ecs.prefab<prefabs::Cannon::Head>()
            .is_a<prefabs::materials::CannonHead>()
            .set<Box>({0.8, 0.4, 0.8})
            .set<Position>({0, -0.8, 0})
            .set<Rotation>({0, 0.0, 0});

            ecs.prefab<prefabs::Cannon::Barrel>()
                .is_a<prefabs::materials::Metal>()
                .set<Box>({0.8, 0.14, 0.14});

            ecs.prefab<prefabs::Cannon::Head::BarrelLeft>()
                .slot_of<prefabs::Cannon>()
                .is_a<prefabs::Cannon::Barrel>()
                .set<Position>({TurretCannonLength, 0.0, -TurretCannonOffset}); 

            ecs.prefab<prefabs::Cannon::Head::BarrelRight>()
                .slot_of<prefabs::Cannon>()
                .is_a<prefabs::Cannon::Barrel>()
                .set<Position>({TurretCannonLength, 0.0, TurretCannonOffset});                         

    ecs.prefab<prefabs::Laser>().is_a<::prefabs::Turret>()
        .add<Laser>()
        .set<Turret>({BeamFireInterval});

        ecs.prefab<prefabs::Laser::Head>()
            .set<Position>({0.0, -0.8, 0})
            .set<Rotation>({0, 0.0, 0});

            ecs.prefab().is_a<prefabs::materials::Metal>()
                .child_of<prefabs::Laser::Head>()
                .set<Box>({0.9, 0.3, 0.16})
                .set<Position>({0.1, 0.0, -0.3});
                
            ecs.prefab().is_a<prefabs::materials::Metal>()
                .child_of<prefabs::Laser::Head>()
                .set<Box>({0.9, 0.3, 0.16})
                .set<Position>({0.1, 0.0, 0.3});                

            ecs.prefab().is_a<prefabs::materials::LaserLight>()
                .child_of<prefabs::Laser::Head>()
                .set<Box>({0.8, 0.2, 0.1})
                .set<Position>({0.1, 0.0, -0.20});

            ecs.prefab().is_a<prefabs::materials::LaserLight>()
                .child_of<prefabs::Laser::Head>()
                .set<Box>({0.8, 0.2, 0.1})
                .set<Position>({0.1, 0.0, 0.2});

            ecs.prefab().is_a<prefabs::materials::Metal>()
                .child_of<prefabs::Laser::Head>()
                .set<Box>({1.6, 0.5, 0.3})
                .set<Position>({0.24, 0, 0.0});  

            ecs.prefab().is_a<prefabs::materials::Metal>()
                .child_of<prefabs::Laser::Head>()
                .set<Box>({1.0, 0.16, 0.16})
                .set<Position>({0.8, -0.1, 0.0});                  

        ecs.prefab<prefabs::Laser::Head::Beam>().slot_of<prefabs::Laser>()
            .is_a<prefabs::materials::Beam>()
            .set<Position>({2, 0, 0})
            .set<Box>({0}).override<Box>()
            .set<Rotation>({0,(float)GLM_PI / 2.0f, 0});
}

// Init systems
void init_systems(flecs::world& ecs) {
    ecs.scope<systems>([&](){ // Keep root scope clean

    // Spawn enemies periodically
    ecs.system<const Game>("SpawnEnemy")
        .term_at(1).singleton()
        .interval(EnemySpawnInterval)
        .iter(SpawnEnemy);

    // Move enemies
    ecs.system<Position, Direction, const Game>("MoveEnemy")
        .term_at(3).singleton()
        .term<Enemy>()
        .each(MoveEnemy);

    // Clear invalid target for turrets
    ecs.system<Target, Position>("ClearTarget")
        .each(ClearTarget);

    // Find target for turrets
    ecs.system<Turret, Target, Position>("FindTarget")
        .term<SpatialQuery, Enemy>().up()
        .term<SpatialQueryResult, Enemy>()
        .each(FindTarget);

    // Aim turret at enemies
    ecs.system<Turret, Target, Position>("AimTarget")
        .each(AimTarget);

    // Countdown until next fire
    ecs.system<Turret, Target>("FireCountdown")
        .each(FireCountdown);

    // Aim beam at target
    ecs.system<Position, Turret, Target>("BeamControl")
        .each(BeamControl);

    // Fire bullets at enemies
    ecs.system<Turret, Target, Position>("FireAtTarget")
        .term<Laser>().optional()
        .each(FireAtTarget);

    // Apply recoil to barrels
    ecs.system<Position, const Recoil>("ApplyRecoil")
        .each(ApplyRecoil);

    // Decrease recoil amount over time
    ecs.system<Recoil>("DecreaseRecoil")
        .each(DecreaseRecoil);

    // Decrease recoil amount over time
    ecs.system<HitCooldown>("DecreaseHitCooldown")
        .each(DecreaseHitCoolDown);

    // Simple particle system
    ecs.system<ParticleLifespan, const Particle, Box, Color, Velocity>
            ("ProgressParticle")
        .term_at(2).up() // shared particle properties
        .term_at(3).optional()
        .term_at(4).optional()
        .term_at(5).optional()
        .instanced()
        .iter(ProgressParticle);

    // Test for collisions with enemies
    ecs.system<Position, Health, Box, HitCooldown>("HitTarget")
        .term<SpatialQuery, Bullet>().up()
        .term<SpatialQueryResult, Bullet>()
        .each(HitTarget);

    // Destroy enemy when health goes to 0
    ecs.system<Health, Position>("DestroyEnemy")
        .term<Enemy>()
        .each(DestroyEnemy);

    // Update enemy color when hit & when health decreases
    ecs.system<Color, Health, const HitCooldown>("UpdateEnemyColor")
        .each(UpdateEnemyColor);
    });
}

int main(int argc, char *argv[]) {
    flecs::world ecs(argc, argv);

    flecs::log::set_level(0);

    ecs.import<flecs::components::transform>();
    ecs.import<flecs::components::graphics>();
    ecs.import<flecs::components::geometry>();
    ecs.import<flecs::components::gui>();
    ecs.import<flecs::components::physics>();
    ecs.import<flecs::components::input>();
    ecs.import<flecs::systems::transform>();
    ecs.import<flecs::systems::physics>();
    ecs.import<flecs::monitor>();
    ecs.import<flecs::game>();
    ecs.import<flecs::systems::sokol>();

    init_game(ecs);
    init_ui(ecs);
    init_prefabs(ecs);
    init_level(ecs);
    init_systems(ecs);

    ecs.app()
        .enable_rest()
        .run();
}
