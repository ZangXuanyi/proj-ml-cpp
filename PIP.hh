#include <array>
#include <vector>
#include <ranges>
#include <iostream>
#include <print>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <format>
#include <limits>

namespace pip {

struct GAConfig
{
    std::size_t population_size = 80;
    std::size_t elite_count = 4;
    std::size_t max_points = 0;
    double count_weight = 10000.0;
    double min_dist2_weight = 100.0;
    double penalty_weight_start = 5.0;
    double penalty_weight_end = 50.0;
    double target_dist2 = 1.0;
    double mutation_rate = 0.15;
    double mutation_sigma = 0.03;
    double insert_rate = 0.25;
    std::size_t insert_candidates = 200;
    double delete_rate = 0.05;
    std::size_t repair_steps = 30;
    double repair_step = 0.02;
    double crossover_keep_rate = 0.5;
    double add_point_bonus_rate = 0.15;
};

struct FitnessStats
{
    double min_dist2 = 0.0;
    double penalty = 0.0;
    double score = 0.0;
};

template <std::size_t N>
class Point // N维空间中的点，最好保证在球面上
{
private:
    std::array<double, N> coords;

public:
    Point(const std::array<double, N> &arr) : coords(arr) {} // 用于从数组直接构造点
    Point()                                                  // 随机生成一个在单位球面上的点，用于之后的变异，不用于初始化
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        std::ranges::generate(
            coords,
            [&]()
            { return dis(gen); });
        // 归一化到单位球面上
        double norm_squared = std::accumulate(
            coords.begin(), coords.end(), 0.0,
            [](double sum, double val)
            { return sum + val * val; });
        double norm = std::sqrt(norm_squared);
        for (auto &c : coords)
        {
            c /= norm;
        }
    }

    static Point randomOnSphere(std::mt19937 &gen)
    {
        std::normal_distribution<double> dist(0.0, 1.0);
        std::array<double, N> arr{};
        double norm_squared = 0.0;
        for (std::size_t i = 0; i < N; ++i)
        {
            arr[i] = dist(gen);
            norm_squared += arr[i] * arr[i];
        }
        double inv_norm = 1.0 / std::sqrt(norm_squared);
        for (auto &v : arr)
        {
            v *= inv_norm;
        }
        return Point(arr);
    }

    void normalize()
    {
        double norm_squared = 0.0;
        for (double v : coords)
        {
            norm_squared += v * v;
        }
        if (norm_squared == 0.0)
        {
            coords.fill(0.0);
            coords[0] = 1.0;
            return;
        }
        double inv_norm = 1.0 / std::sqrt(norm_squared);
        for (auto &v : coords)
        {
            v *= inv_norm;
        }
    }

    void jitter(double sigma, std::mt19937 &gen)
    {
        if (sigma <= 0.0)
        {
            return;
        }
        std::normal_distribution<double> noise(0.0, sigma);
        for (auto &v : coords)
        {
            v += noise(gen);
        }
        normalize();
    }

    const std::array<double, N> &getCoords() const
    {
        return coords;
    }

    double squareDistanceTo(const Point<N> &other) const
    {
        double sum = 0.0;
        for (std::size_t i = 0; i < N; ++i)
        {
            double diff = coords[i] - other.coords[i];
            sum += diff * diff;
        }
        return sum;
    }

    static std::vector<Point<N>> generateRandomPoints(std::size_t count) // 生成 count 个随机点
    {
        std::vector<Point<N>> points(count);
        for (auto &p : points)
        {
            p = Point<N>();
        }
        return points;
    }

    // 转成字符串的方法，便于打印
    std::string toString() const
    {
        std::string out = "(";
        for (std::size_t i = 0; i < coords.size(); ++i)
        {
            if (i) out += ", ";
            out += std::format("{:.3f}", coords[i]);
        }
        out += ")";
        return out;
    }

};

template <std::size_t N>
class Individual // 用于遗传算法的“个体”，也就是一组点
{
private:
    std::vector<Point<N>> points;

public:
    Individual(const std::vector<Point<N>> &pts) : points(pts) {} // 手动构造个体

    std::vector<Point<N>> &getPoints()
    {
        return points;
    }

    const std::vector<Point<N>> &getPoints() const
    {
        return points;
    }

    std::size_t getPointCount() const
    {
        return points.size();
    }

    bool isValid() const // 这些点实际上是一些球的球心，球的直径是1（记一开始小球的半径为1/2故而这些球心全部落在单位超球面上）
    // 要求任意两个球心之间的距离至少为1，才能保证这些球不会相互重叠
    {
        std::size_t sz = points.size();
        for (std::size_t i = 0; i < sz; ++i)
            for (std::size_t j = i + 1; j < sz; ++j)
                if (points[i].squareDistanceTo(points[j]) < 1) // 我是大笨蛋，一开始竟然写的4
                    return false;
        return true;
    }

    FitnessStats evaluate(const GAConfig &config, double penalty_weight) const
    {
        FitnessStats stats;
        if (points.size() < 2)
        {
            stats.min_dist2 = 0.0;
            stats.penalty = 0.0;
            stats.score = config.count_weight * static_cast<double>(points.size());
            return stats;
        }

        stats.min_dist2 = std::numeric_limits<double>::infinity();
        std::size_t sz = points.size();
        for (std::size_t i = 0; i < sz; ++i)
        {
            for (std::size_t j = i + 1; j < sz; ++j)
            {
                double d2 = points[i].squareDistanceTo(points[j]);
                stats.min_dist2 = std::min(stats.min_dist2, d2);
                if (d2 < config.target_dist2)
                {
                    double gap = config.target_dist2 - d2;
                    stats.penalty += gap * gap;
                }
            }
        }

        stats.score = config.count_weight * static_cast<double>(points.size()) +
                      config.min_dist2_weight * stats.min_dist2 -
                      penalty_weight * stats.penalty;
        return stats;
    }

    static std::size_t worst_point_index(const std::vector<Point<N>> &pts)
    {
        std::size_t n = pts.size();
        if (n == 0)
        {
            return 0;
        }
        std::vector<double> min_d2(n, std::numeric_limits<double>::infinity());
        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t j = i + 1; j < n; ++j)
            {
                double d2 = pts[i].squareDistanceTo(pts[j]);
                if (d2 < min_d2[i])
                {
                    min_d2[i] = d2;
                }
                if (d2 < min_d2[j])
                {
                    min_d2[j] = d2;
                }
            }
        }
        return static_cast<std::size_t>(std::min_element(min_d2.begin(), min_d2.end()) - min_d2.begin());
    }

    static Point<N> best_insert_candidate(const std::vector<Point<N>> &pts,
                                          std::size_t candidates,
                                          std::mt19937 &gen)
    {
        Point<N> best = Point<N>::randomOnSphere(gen);
        double best_min = -1.0;
        for (std::size_t i = 0; i < candidates; ++i)
        {
            Point<N> candidate = Point<N>::randomOnSphere(gen);
            double min_d2 = std::numeric_limits<double>::infinity();
            for (const auto &p : pts)
            {
                min_d2 = std::min(min_d2, candidate.squareDistanceTo(p));
            }
            if (min_d2 > best_min)
            {
                best_min = min_d2;
                best = candidate;
            }
        }
        return best;
    }

    void prune_to_size(std::size_t target_size)
    {
        while (points.size() > target_size && points.size() > 1)
        {
            std::size_t idx = worst_point_index(points);
            points.erase(points.begin() + idx);
        }
    }

    void repair(const GAConfig &config)
    {
        if (points.size() < 2 || config.repair_steps == 0)
        {
            return;
        }

        std::vector<std::array<double, N>> forces(points.size());
        for (std::size_t step = 0; step < config.repair_steps; ++step)
        {
            for (auto &f : forces)
            {
                f.fill(0.0);
            }
            for (std::size_t i = 0; i < points.size(); ++i)
            {
                for (std::size_t j = i + 1; j < points.size(); ++j)
                {
                    double d2 = points[i].squareDistanceTo(points[j]);
                    if (d2 >= config.target_dist2)
                    {
                        continue;
                    }
                    double gap = config.target_dist2 - d2;
                    double dist = std::sqrt(d2) + 1e-12;
                    double scale = gap / dist;
                    const auto &ci = points[i].getCoords();
                    const auto &cj = points[j].getCoords();
                    for (std::size_t k = 0; k < N; ++k)
                    {
                        double diff = ci[k] - cj[k];
                        double push = diff * scale;
                        forces[i][k] += push;
                        forces[j][k] -= push;
                    }
                }
            }
            for (std::size_t i = 0; i < points.size(); ++i)
            {
                auto coords = points[i].getCoords();
                for (std::size_t k = 0; k < N; ++k)
                {
                    coords[k] += config.repair_step * forces[i][k];
                }
                points[i] = Point<N>(coords);
                points[i].normalize();
            }
        }
    }

    void mutate(const GAConfig &config, std::mt19937 &gen)
    {
        if (points.empty())
        {
            return;
        }

        std::uniform_real_distribution<double> uni(0.0, 1.0);

        for (auto &p : points)
        {
            if (uni(gen) < config.mutation_rate)
            {
                p.jitter(config.mutation_sigma, gen);
            }
        }

        if (uni(gen) < config.delete_rate && points.size() > 2)
        {
            std::size_t idx = worst_point_index(points);
            points.erase(points.begin() + idx);
        }

        if (uni(gen) < config.insert_rate)
        {
            Point<N> candidate = best_insert_candidate(points, config.insert_candidates, gen);
            points.push_back(candidate);
        }
    }

    static Individual crossover(const Individual &a,
                                const Individual &b,
                                const GAConfig &config,
                                std::mt19937 &gen)
    {
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::vector<Point<N>> child;
        child.reserve(a.points.size() + b.points.size());

        for (const auto &p : a.points)
        {
            if (uni(gen) < config.crossover_keep_rate)
            {
                child.push_back(p);
            }
        }
        for (const auto &p : b.points)
        {
            if (uni(gen) < config.crossover_keep_rate)
            {
                child.push_back(p);
            }
        }
        if (child.empty())
        {
            child.push_back(a.points.front());
        }

        std::size_t target = config.max_points;
        if (target == 0)
        {
            target = std::max(a.points.size(), b.points.size());
        }
        if (uni(gen) < config.add_point_bonus_rate)
        {
            target += 1;
        }
        if (child.size() > target)
        {
            Individual temp(child);
            temp.prune_to_size(target);
            return temp;
        }
        return Individual(child);
    }

    std::string toString() const
    {
        std::string out = "[";
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            if (i) out += ", ";
            out += points[i].toString();
        }
        out += "]";
        return out;
    }

};

template <std::size_t N>
class Population // 用于遗传算法的“种群”，包含多个个体
{
private:
    std::vector<Individual<N>> individuals;

public:
    Population(std::size_t populationSize,
               const std::vector<Point<N>> &seed,
               const GAConfig &config,
               std::mt19937 &gen)
    {
        individuals.clear();
        individuals.reserve(populationSize);
        if (!seed.empty())
        {
            individuals.emplace_back(seed);
        }
        while (individuals.size() < populationSize)
        {
            Individual<N> ind(seed);
            ind.mutate(config, gen);
            ind.repair(config);
            individuals.push_back(std::move(ind));
        }
    }

    const Individual<N> &getBestIndividual(const GAConfig &config, double penalty_weight) const
    {
        return *std::max_element(
            individuals.begin(), individuals.end(),
            [&](const Individual<N> &a, const Individual<N> &b)
            {
                return a.evaluate(config, penalty_weight).score <
                       b.evaluate(config, penalty_weight).score;
            });
    }

    void evolve(std::size_t generations, const GAConfig &config, std::mt19937 &gen)
    {
        if (individuals.empty())
        {
            return;
        }

        for (std::size_t genIdx = 0; genIdx < generations; ++genIdx)
        {
            double t = (generations > 1)
                           ? static_cast<double>(genIdx) / static_cast<double>(generations - 1)
                           : 1.0;
            double penalty_weight = config.penalty_weight_start +
                                    (config.penalty_weight_end - config.penalty_weight_start) * t;

            std::vector<std::pair<double, Individual<N>>> ranked;
            ranked.reserve(individuals.size());
            for (const auto &ind : individuals)
            {
                ranked.emplace_back(ind.evaluate(config, penalty_weight).score, ind);
            }
            std::sort(ranked.begin(), ranked.end(),
                      [](const auto &a, const auto &b)
                      { return a.first > b.first; });

            std::vector<Individual<N>> next;
            std::size_t elite = std::min(config.elite_count, ranked.size());
            for (std::size_t i = 0; i < elite; ++i)
            {
                next.push_back(ranked[i].second);
            }

            std::uniform_int_distribution<std::size_t> parent_dis(0, ranked.size() / 2);
            while (next.size() < individuals.size())
            {
                const Individual<N> &p1 = ranked[parent_dis(gen)].second;
                const Individual<N> &p2 = ranked[parent_dis(gen)].second;
                Individual<N> child = Individual<N>::crossover(p1, p2, config, gen);
                child.mutate(config, gen);
                child.repair(config);
                next.push_back(std::move(child));
            }

            individuals = std::move(next);
        }
    }
};

} // namespace pip
