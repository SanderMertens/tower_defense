#include <iostream>
#include <initializer_list>
#include <tower_defense.h>
#include <vector>

using namespace std;
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
using PointLight = graphics::PointLight;

#define ECS_PI_2 ((float)(GLM_PI * 2))

// Game constants
static const int LevelScale = 1;
static const float EnemySpeed = 5.0;
static const float EnemySpawnInterval = 0.15;

static const float RecoilAmount = 0.3;
static const float DecreaseRecoilRate = 1.5;
static const float HitCooldownRate = 1.0;
static const float HitCooldownInitialValue = 0.25;

static const float TurretRotateSpeed = 4.0;
static const float TurretRange = 5.0;
static const float TurretCannonOffset = 0.2;
static const float TurretCannonLength = 0.6;

static const float BulletSpeed = 26.0;
static const float BulletDamage = 0.015;

static const float BeamDamage = 0.25;
static const float BeamSize = 0.06;

static const float SmokeSize = 1.5;
static const float ExplodeRadius = 1.0;
static const int SmokeParticleCount = 50;

static const float SparkInitialVelocity = 11.0;
static const int SparkParticleCount = 30;
static const float SparkSize = 0.15;

static const float TileSize = 3.0;
static const float TileHeight = 0.6;
static const float TileSpacing = 0.00;
static const int TileCountX = 20;
static const int TileCountZ = 20;

// Direction vector. During pathfinding enemies will cycle through this vector
// to find the next direction to turn to.
static const transform::Position2 dir[] = {
    {-1, 0},
    {0, -1},
    {1, 0},
    {0, 1},
};

// Level builder utilities
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

struct Waypoint {
    float x, y;
};

enum class TileKind {
    Turret = 0, // Default
    Path,
    Other
};

struct Waypoints {
    Waypoints(grid<TileKind> *g, initializer_list<Waypoint> pts) : tiles(g) {
        for (const auto& p : pts)
            add(p, TileKind::Path);
    }

    void add(Waypoint next, TileKind kind) {
        next.x *= LevelScale; next.y *= LevelScale;
        if (next.x == last.x) {
            do {
                last.y += (last.y < next.y) - (last.y > next.y);
                tiles->set(last.x, last.y, kind);
            } while (next.y != last.y);
        } else if (next.y == last.y) {
            do {
                last.x += (last.x < next.x) - (last.x > next.x);
                tiles->set(last.x, last.y, kind);
            } while (next.x != last.x);
        }

        last.x = next.x;
        last.y = next.y;
    }

    void fromTo(Waypoint first, Waypoint second, TileKind kind) {
        last = first;
        add(second, kind);
    }

    grid<TileKind> *tiles = nullptr;
    Waypoint last = {0, 0};
};

// Components

namespace tower_defense {

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

    Level(grid<TileKind> *arg_map, transform::Position2 arg_spawn) {
        map = arg_map;
        spawn_point = arg_spawn;
    }

    grid<TileKind> *map;
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

struct ExplosionLight {
    float intensity;
    float decay;
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
        float height;
        float variation;
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
    };

    struct Laser {
        struct Head {
            struct Beam { };
        };
    };
}

}

using namespace tower_defense;

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

float toX(float x) {
    return to_coord(x + 0.5) - to_coord((TileCountX / 2.0));
}

float toZ(float z) {
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
bool find_path(Position& p, Direction& d, const Level& lvl) {
    // Check if enemy is in center of tile
    float t_x = from_x(p.x);
    float t_y = from_z(p.z);
    int ti_x = (int)t_x;
    int ti_y = (int)t_y;
    float td_x = t_x - ti_x;
    float td_y = t_y - ti_y;

    // If enemy is in center of tile, decide where to go next
    if (td_x < 0.1 && td_y < 0.1) {
        grid<TileKind> *tiles = lvl.map;

        // Compute backwards direction so we won't try to go there
        int backwards = (d.value + 2) % 4;

        // Find a direction that the enemy can move to
        for (int i = 0; i < 3; i ++) {
            int n_x = ti_x + dir[d.value].x;
            int n_y = ti_y + dir[d.value].y;

            if (n_x >= 0 && n_x <= TileCountX) {
                if (n_y >= 0 && n_y <= TileCountZ) {
                    // Next tile is still on the grid, test if it's a path
                    if (tiles[0](n_x, n_y) == TileKind::Path) {
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

void SpawnEnemy(flecs::iter& it, size_t, const Game& g) {
    const Level& lvl = g.level.get<Level>();

    it.world().entity().child_of<enemies>().is_a<prefabs::Enemy>()
        .set<Direction>({0})
        .set<Position>({
            lvl.spawn_point.x, 1.2, lvl.spawn_point.y
        });
}

void MoveEnemy(flecs::iter& it, size_t i,
    Position& p, Direction& d, const Game& g)
{
    const Level& lvl = g.level.get<Level>();

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
            Position target_pos = t.get<Position>();
            float distance = glm_vec3_distance(p, target_pos);
            if (distance > TurretRange) {
                // Target is out of range
                target.target = flecs::entity::null();
                target.lock = false;
            }
        }
    }
}

void FindTarget(flecs::iter& it, size_t i, Turret& turret, Target& target, 
    Position& p, const SpatialQuery& q, SpatialQueryResult& qr) 
{
    if (target.target) {
        // Already has a target
        return;
    }

    flecs::entity enemy;
    float distance = 0, min_distance = 0;

    // Find all enemies around the turret's position within TurretRange
    q.findn(p, TurretRange, qr);
    for (auto e : qr) {
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

        Position target_p = enemy.get<Position>();
        vec3 diff;
        glm_vec3_sub(target_p, target.prev_position, diff);

        target.prev_position[0] = target_p.x;
        target.prev_position[1] = target_p.y;
        target.prev_position[2] = target_p.z;
        float distance = glm_vec3_distance(p, target_p);

        // Crude correction for enemy movement and bullet travel time
        flecs::entity beam = e.target<prefabs::Laser::Head::Beam>();
        if (!beam) {
            glm_vec3_scale(diff, distance * 1, diff);
            glm_vec3_add(target_p, diff, target_p);
        }

        target.aim_position[0] = target_p.x;
        target.aim_position[1] = target_p.y;
        target.aim_position[2] = target_p.z;            

        float angle = look_at(p, target_p);

        flecs::entity head = e.target<prefabs::Turret::Head>();
        Rotation r = head.get<Rotation>();
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
    bool is_laser = it.is_set(3);
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
            pos.x += sin(angle) * 0.8 * TurretCannonOffset * turret.lr;
            pos.y = 1.1;
            pos.z += cos(angle) * 0.8 * TurretCannonOffset * turret.lr;

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
                .child_of<particles>()
                .set<Position>(pos)
                .set<Velocity>({-v[0], 0, -v[2]});
            ecs.entity().is_a<prefabs::NozzleFlash>()
                .child_of<particles>()
                .set<Position>(pos)
                .set<Rotation>({0, angle, 0});

            // Create nozzle flash light
            ecs.entity()
                .child_of<particles>()
                .set<Position>({pos.x, pos.y, pos.z})
                .set<PointLight>({{0.5, 0.4, 0.2}, 0.4})
                .set<ExplosionLight>({1.0, 7.0}); 
        } else {
            // Enable laser beam
            e.target<prefabs::Laser::Head::Beam>().enable();
            pos.x += 1.4 * -v[0];
            pos.y = 1.1;
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
        if (beam.enabled()) {
            // Disable beam if laser turret has no target
            beam.disable();
            beam.set<Box>({0.0, 0.0, 0});
        }
        return;
    }

    if (target.lock && beam && beam.enabled()) {
        flecs::entity enemy = target.target;
        if (!enemy.is_alive()) {
            return;
        }

        // Position beam at enemy
        Position target_pos = enemy.get<Position>();
        float distance = glm_vec3_distance(p, target_pos);
        beam.set<Position>({ (distance / 2), 0.1, 0.0 });
        beam.set<Box>({BeamSize, BeamSize, distance});

        // Subtract health from enemy as long as beam is firing
        enemy.get([&](Health& h, HitCooldown& hc) {
            h.value -= BeamDamage * it.delta_time();
            hc.value = HitCooldownInitialValue;
        });

        // Generate spark   
        {     
            float x_r = randf(ECS_PI_2);
            float y_r = randf(ECS_PI_2);
            float z_r = randf(ECS_PI_2);
            float speed = randf(5) + 2.0;
            float size = randf(0.15);

            it.world().scope<particles>().entity().is_a<prefabs::Ion>()
                .child_of<particles>()
                .set<Position>({target_pos.x, target_pos.y, target_pos.z}) 
                .set<Box>({size, size, size})
                .set<Velocity>({
                    cos(x_r) * speed, fabs(cos(y_r) * speed), cos(z_r) * speed});
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

void ProgressParticle(flecs::iter& it, size_t i,
    ParticleLifespan& pl, const Particle& p, Box *box, Color *color, Velocity *vel)
{
    if (box) {
        box->width *= pow(p.size_decay, it.delta_time());
        box->height *= pow(p.size_decay, it.delta_time());
        box->depth *= pow(p.size_decay, it.delta_time());
    }
    if (color) {
        color->r *= pow(p.color_decay, it.delta_time());
        color->g *= pow(p.color_decay, it.delta_time());
        color->b *= pow(p.color_decay, it.delta_time()); 
    }
    if (vel) {
        vel->x *= pow(p.velocity_decay, it.delta_time());
        vel->y *= pow(p.velocity_decay, it.delta_time());
        vel->z *= pow(p.velocity_decay, it.delta_time());
    }

    pl.t += it.delta_time();
    if (pl.t > p.lifespan || ((box->width + box->height + box->depth) < 0.1)) {
        it.entity(i).destruct();
    }
}

void explode(flecs::world& ecs, Position& p, float pC, float rC, Color rgbRnd, Color rgbC) {
    // Create explosion particles that fade into smoke
    for (int s = 0; s < SmokeParticleCount * pC; s ++) {
        float red = randf(rgbRnd.r) + rgbC.r;
        float green = randf(rgbRnd.g) + rgbC.g;
        float blue = randf(rgbRnd.b) + rgbC.b;
        float size = SmokeSize * randf(1.0) * rC;

        Position pp;
        pp.x = p.x + randf(ExplodeRadius) - ExplodeRadius / 2; 
        pp.y = p.y + randf(ExplodeRadius) - ExplodeRadius / 2;
        pp.z = p.z + randf(ExplodeRadius) - ExplodeRadius / 2;

        ecs.scope<particles>().entity().is_a<prefabs::Smoke>()
            .set<Position>(pp)
            .set<Box>({size, size, size})
            .set<Color>({red, green, blue});
    }

    // Create sparks
    for (int s = 0; s < SparkParticleCount * pC; s ++) {
        float x_r = randf(ECS_PI_2);
        float y_r = randf(ECS_PI_2);
        float z_r = randf(ECS_PI_2);
        float speed = randf(SparkInitialVelocity) * rC + 2.0;
        float size = SparkSize + randf(0.2);

        ecs.scope<particles>().entity().is_a<prefabs::Spark>()
            .set<Position>({p.x, p.y, p.z}) 
            .set<Box>({size, size, size})
            .set<Velocity>({
                cos(x_r) * speed, fabs(cos(y_r) * speed), cos(z_r) * speed});
    }

    // Create explosion light
    ecs.entity()
        .child_of<particles>()
        .set<Position>({p.x, p.y, p.z})
        .set<PointLight>({{rgbC.r, rgbC.g, rgbC.b}, 1.25f + pC / 2.0f})
        .set<ExplosionLight>({0.75f * (0.5f + pC / 2.0f), 1.5f});
}

void HitTarget(flecs::iter& it, size_t i, Position& p, Health& h, Box& b, 
    HitCooldown& hit_cooldown, const SpatialQuery& q, SpatialQueryResult& qr)
{    
    flecs::world ecs = it.world();
    flecs::entity enemy = it.entity(i);
    float range = b.width / 2;

    // Test whether bullet hit an enemy
    q.findn(p, range, qr);
    for (auto e : qr) {
        it.world().entity(e.id).destruct();
        auto prevHealth = h.value;
        h.value -= BulletDamage;
        if (prevHealth > 0.9 && h.value < 0.9) {
            explode(ecs, p, 0.2, 0.3, {0.01, 0.3, 0.3}, {0.05, 0.7, 0.2});
            enemy.set<Color>({0.05, 0.2, 0.6});
        } else if (prevHealth > 0.7 && h.value < 0.7) {
            explode(ecs, p, 0.4, 0.5, {0.01, 0.3, 0.3}, {0.01, 0.2, 0.8});
            enemy.set<Color>({0.2, 0.05, 0.4});
        } else if (prevHealth > 0.5 && h.value < 0.5) {
            explode(ecs, p, 0.5, 0.5, {0.3, 0.01, 0.3}, {0.01, 0.01, 0.7});
            enemy.set<Color>({0.2, 0.05, 0.2});
        } else if (prevHealth > 0.3 && h.value < 0.3) {
            explode(ecs, p, 0.6, 0.7, {0.5, 0.2, 0.5}, {0.8, 0.01, 0.8});
            enemy.set<Color>({0.1, 0.03, 0.0});
        }
        hit_cooldown.value = HitCooldownInitialValue; // For color effect
    }
}

void DestroyEnemy(flecs::entity e, Health& h, Position& p) {
    flecs::world ecs = e.world();
    if (h.value <= 0) {
        e.destruct();
        explode(ecs, p, 1.1, 1.0, {0.5, 0.2, 0.1}, {0.7, 0.1, 0.05});
    }
}

void init_components(flecs::world& ecs) {
    ecs.component<Game>()
        .member("window", &Game::window)
        .member("level", &Game::window)
        .member("center", &Game::center)
        .member("size", &Game::size);

    ecs.component<Enemy>();
    ecs.component<Laser>();
    ecs.component<Bullet>();

    ecs.component<Particle>()
        .member("size_decay", &Particle::size_decay)
        .member("color_decay", &Particle::color_decay)
        .member("velocity_decay", &Particle::velocity_decay)
        .member("lifespan", &Particle::lifespan)
        .add(flecs::OnInstantiate, flecs::Inherit);

    ecs.component<ParticleLifespan>();

    ecs.component<Health>()
        .member("value", &Health::value);
    
    ecs.component<HitCooldown>()
        .member("value", &HitCooldown::value);

    ecs.component<Turret>()
        .member("fire_interval", &Turret::fire_interval);

    ecs.component<Target>()
        .member("target", &Target::target)
        .member("prev_position", &Target::prev_position)
        .member("aim_position", &Target::aim_position)
        .member("angle", &Target::angle)
        .member("distance", &Target::distance)
        .member("lock", &Target::lock);
}

void init_game(flecs::world& ecs) {
    // Singleton with global game data
    ecs.component<Game>().add(flecs::Singleton);

    Game& g = ecs.ensure<Game>();
    g.center = { toX(TileCountX / 2), 0, toZ(TileCountZ / 2) };
    g.size = TileCountX * (TileSize + TileSpacing) + 2;

    // Camera, lighting & canvas configuration
    ecs.script().filename("etc/assets/app.flecs").run();

    // Prefab assets
    ecs.script().filename("etc/assets/materials.flecs").run();
    ecs.script().filename("etc/assets/tree.flecs").run();
    ecs.script().filename("etc/assets/tile.flecs").run();
    ecs.script().filename("etc/assets/particle.flecs").run();
    ecs.script().filename("etc/assets/bullet.flecs").run();
    ecs.script().filename("etc/assets/nozzle_flash.flecs").run();
    ecs.script().filename("etc/assets/smoke.flecs").run();
    ecs.script().filename("etc/assets/spark.flecs").run();
    ecs.script().filename("etc/assets/ion.flecs").run();
    ecs.script().filename("etc/assets/bolt.flecs").run();
    ecs.script().filename("etc/assets/enemy.flecs").run();
    ecs.script().filename("etc/assets/turret.flecs").run();
    ecs.script().filename("etc/assets/cannon.flecs").run();
    ecs.script().filename("etc/assets/laser.flecs").run();
}

// Build level
void init_level(flecs::world& ecs) {
    Game& g = ecs.ensure<Game>();

    grid<TileKind> *path = new grid<TileKind>(
        TileCountX * LevelScale, TileCountZ * LevelScale);

    Waypoints waypoints(path, {
        {0, 1}, {8, 1}, {8, 3}, {1, 3}, {1, 8}, {4, 8}, {4, 5}, {8, 5}, {8, 7},
        {6, 7}, {6, 9}, {11, 9}, {11, 1}, {18, 1}, {18, 3}, {16, 3}, {16, 5},
        {18, 5}, {18, 7}, {16, 7}, {16, 9}, {18, 9}, {18, 12}, {1, 12}, {1, 18},
        {3, 18}, {3, 15}, {5, 15}, {5, 18}, {7, 18}, {7, 15}, {9, 15}, {9, 18},
        {12, 18}, {12, 14}, {18, 14}, {18, 16}, {14, 16}, {14, 19}, {19, 19}
    });

    transform::Position2 spawn_point = {
        toX(LevelScale * TileCountX - 1), 
        toZ(LevelScale * TileCountZ - 1)
    };

    g.level = ecs.entity()
        .child_of<Level>()
        .set<Level>({path, spawn_point});

    ecs.entity("GroundPlane")
        .child_of<level>()
        .set<Position>({0, -2.7, toZ(TileCountZ / 2 - 0.5)})
        .set<Box>({toX(TileCountX + 0.5) * 20, 5, toZ(TileCountZ + 2) * 10})
        .set<Color>({0.11, 0.15, 0.1});

    for (int x = 0; x < TileCountX * LevelScale; x ++) {
        for (int z = 0; z < TileCountZ * LevelScale; z++) {
            float xc = toX(x);
            float zc = toZ(z);

            auto t = ecs.scope<level>().entity().set<Position>({xc, 0, zc});
            if (path[0](x, z) == TileKind::Path) {
                t.is_a<prefabs::Path>();
            } else if (path[0](x, z) == TileKind::Turret) {
                t.is_a<prefabs::Tile>();

                bool canTurret = false;
                if (x < (TileCountX * LevelScale - 1) && (z < (TileCountZ * LevelScale - 1))) {
                    canTurret |= (path[0](x + 1, z) == TileKind::Path);
                    canTurret |= (path[0](x, z + 1) == TileKind::Path);
                }
                if (x && z) {
                    canTurret |= (path[0](x - 1, z) == TileKind::Path);
                    canTurret |= (path[0](x, z - 1) == TileKind::Path);
                }

                auto e = ecs.entity().set<Position>({xc, TileHeight / 2, zc});
                if (!canTurret || (randf(1) > 0.3)) {
                    if (randf(1) > 0.05) {
                        e.child_of<level>();
                        e.set<prefabs::Tree>({
                            1.5f + randf(2.5),
                            randf(0.1)
                        });
                        e.set<Rotation>({0, randf(2.0 * M_PI)});
                    } else {
                        e.destruct();
                    }
                } else {
                    e.child_of<turrets>();
                    if (randf(1) > 0.3) {
                        e.is_a<prefabs::Cannon>();
                    } else {
                        e.is_a<prefabs::Laser>();
                        e.target<prefabs::Laser::Head::Beam>().disable();
                    }
                }
            } else if (path[0](x, z) == TileKind::Other) {
                t.is_a<prefabs::Tile>();
            }
        }
    }
}

void init_systems(flecs::world& ecs) {
    ecs.scope(ecs.entity("tower_defense"), [&](){ // Keep root scope clean

    // Spawn enemies periodically
    ecs.system<const Game>("SpawnEnemy")
        .interval(EnemySpawnInterval)
        .each(SpawnEnemy);

    // Move enemies
    ecs.system<Position, Direction, const Game>("MoveEnemy")
        .with<Enemy>()
        .each(MoveEnemy);

    // Clear invalid target for turrets
    ecs.system<Target, Position>("ClearTarget")
        .each(ClearTarget);

    // Find target for turrets
    ecs.system<Turret, Target, Position, const SpatialQuery, SpatialQueryResult>
            ("FindTarget")
        .term_at(3).up(flecs::IsA).second<Enemy>() // SpatialQuery(up, Enemy)
        .term_at(4).second<Enemy>()                // (SpatialQueryResult, Enemy)
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
        .with<Laser>().optional()
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
    ecs.system<ParticleLifespan, const Particle, Box*, Color*, Velocity*>
            ("ProgressParticle")
        .term_at(1).up(flecs::IsA) // shared particle properties
        .each(ProgressParticle);

    // Test for collisions with enemies
    ecs.system<Position, Health, Box, HitCooldown, const SpatialQuery, SpatialQueryResult>
            ("HitTarget")
        .term_at(4).up(flecs::IsA).second<Bullet>() // SpatialQuery(up, Bullet)
        .term_at(5).second<Bullet>()                // (SpatialQueryResult, Bullet)
        .each(HitTarget);

    // Destroy enemy when health goes to 0
    ecs.system<Health, Position>("DestroyEnemy")
        .with<Enemy>()
        .each(DestroyEnemy);

    // Decrease intensity of explosion light over time
    ecs.system<ExplosionLight, PointLight>("UpdateExplosionLight")
        .each([](flecs::iter& it, size_t i, ExplosionLight& l, PointLight& p) {
            flecs::entity e = it.entity(i);
            l.intensity -= l.decay * it.delta_time();
            if (l.intensity <= 0) {
                e.destruct();
            } else {
                p.intensity = l.intensity;
            }
        });
    });
}

int main(int argc, char *argv[]) {
    flecs::world ecs(argc, argv);

    ecs.import<flecs::components::transform>();
    ecs.import<flecs::components::graphics>();
    ecs.import<flecs::components::geometry>();
    ecs.import<flecs::components::gui>();
    ecs.import<flecs::components::physics>();
    ecs.import<flecs::components::input>();
    ecs.import<flecs::systems::transform>();
    ecs.import<flecs::systems::physics>();
    ecs.import<flecs::game>();
    ecs.import<flecs::systems::sokol>();

    init_components(ecs);
    init_game(ecs);
    init_level(ecs);
    init_systems(ecs);

    ecs.app()
        .enable_rest()
        .enable_stats()
        .target_fps(60)
        .run();
}
