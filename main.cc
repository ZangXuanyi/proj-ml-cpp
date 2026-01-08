#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using Point = std::vector<double>;
using Points = std::vector<Point>;

struct SearchParams {
    std::size_t dimension = 3;
    std::size_t max_iters = 8000;
    std::size_t restarts = 60;
    std::size_t insert_candidates = 200;
    std::size_t insert_restarts = 20;
    double step_start = 0.06;
    double step_end = 0.005;
    double repulsion = 0.01;
    double penalty_weight = 2.0;
    double tolerance = 1e-6;
    double jitter = 0.0;
    std::size_t jitter_every = 200;
};

static void normalize(Point &p)
{
    double norm_squared = 0.0;
    for (double v : p)
    {
        norm_squared += v * v;
    }
    if (norm_squared == 0.0)
    {
        p.assign(p.size(), 0.0);
        if (!p.empty())
        {
            p[0] = 1.0;
        }
        return;
    }
    double inv_norm = 1.0 / std::sqrt(norm_squared);
    for (double &v : p)
    {
        v *= inv_norm;
    }
}

static Point random_unit_point(std::size_t dim, std::mt19937 &gen)
{
    std::normal_distribution<double> dist(0.0, 1.0);
    Point p(dim);
    for (std::size_t i = 0; i < dim; ++i)
    {
        p[i] = dist(gen);
    }
    normalize(p);
    return p;
}

static Points random_points(std::size_t count, std::size_t dim, std::mt19937 &gen)
{
    Points points;
    points.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        points.push_back(random_unit_point(dim, gen));
    }
    return points;
}

static double distance_squared(const Point &a, const Point &b)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

static double min_distance_squared(const Points &points)
{
    if (points.size() < 2)
    {
        return 0.0;
    }

    double min_dist2 = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        for (std::size_t j = i + 1; j < points.size(); ++j)
        {
            min_dist2 = std::min(min_dist2, distance_squared(points[i], points[j]));
        }
    }
    return min_dist2;
}

static Points generate_dn_roots(std::size_t dim)
{
    Points points;
    if (dim < 2)
    {
        return points;
    }

    points.reserve(dim * (dim - 1) * 2);
    static const int signs[2] = {-1, 1};

    for (std::size_t i = 0; i < dim; ++i)
    {
        for (std::size_t j = i + 1; j < dim; ++j)
        {
            for (int s1 : signs)
            {
                for (int s2 : signs)
                {
                    Point p(dim, 0.0);
                    p[i] = static_cast<double>(s1);
                    p[j] = static_cast<double>(s2);
                    normalize(p);
                    points.push_back(std::move(p));
                }
            }
        }
    }
    return points;
}

static Points generate_e8_roots()
{
    constexpr std::size_t dim = 8;
    Points points;
    points.reserve(240);

    static const int signs[2] = {-1, 1};
    for (std::size_t i = 0; i < dim; ++i)
    {
        for (std::size_t j = i + 1; j < dim; ++j)
        {
            for (int s1 : signs)
            {
                for (int s2 : signs)
                {
                    Point p(dim, 0.0);
                    p[i] = static_cast<double>(s1);
                    p[j] = static_cast<double>(s2);
                    normalize(p);
                    points.push_back(std::move(p));
                }
            }
        }
    }

    for (std::size_t mask = 0; mask < (1u << dim); ++mask)
    {
        Point p(dim, 0.0);
        std::size_t neg_count = 0;
        for (std::size_t d = 0; d < dim; ++d)
        {
            int sign = (mask & (1u << d)) ? -1 : 1;
            if (sign < 0)
            {
                ++neg_count;
            }
            p[d] = 0.5 * static_cast<double>(sign);
        }
        if (neg_count % 2 != 0)
        {
            continue;
        }
        normalize(p);
        points.push_back(std::move(p));
    }

    return points;
}

static Points seed_points_for_dimension(std::size_t dim, std::string &label)
{
    if (dim == 8)
    {
        label = "E8 roots";
        return generate_e8_roots();
    }
    if (dim >= 4)
    {
        label = "D" + std::to_string(dim) + " roots";
        return generate_dn_roots(dim);
    }
    return {};
}

static bool load_points_from_file(const std::string &path, Points &points, std::size_t &out_dim)
{
    std::ifstream input(path);
    if (!input)
    {
        return false;
    }

    std::string line;
    while (std::getline(input, line))
    {
        auto hash_pos = line.find('#');
        if (hash_pos != std::string::npos)
        {
            line = line.substr(0, hash_pos);
        }
        for (char &c : line)
        {
            if (c == ',' || c == ';')
            {
                c = ' ';
            }
        }
        std::istringstream iss(line);
        Point p;
        double value = 0.0;
        while (iss >> value)
        {
            p.push_back(value);
        }
        if (p.empty())
        {
            continue;
        }
        if (out_dim == 0)
        {
            out_dim = p.size();
        }
        if (p.size() != out_dim)
        {
            return false;
        }
        normalize(p);
        points.push_back(std::move(p));
    }

    return !points.empty();
}

static Points insert_point_best(const Points &base, std::size_t candidates, std::mt19937 &gen)
{
    Points result = base;
    if (base.empty())
    {
        return result;
    }

    double best_min_dist2 = -1.0;
    Point best_point;

    for (std::size_t i = 0; i < candidates; ++i)
    {
        Point candidate = random_unit_point(base[0].size(), gen);
        double min_dist2 = std::numeric_limits<double>::infinity();
        for (const auto &p : base)
        {
            min_dist2 = std::min(min_dist2, distance_squared(candidate, p));
        }
        if (min_dist2 > best_min_dist2)
        {
            best_min_dist2 = min_dist2;
            best_point = candidate;
        }
    }

    if (!best_point.empty())
    {
        result.push_back(best_point);
    }
    return result;
}

static bool optimize_points(Points &points, const SearchParams &params, std::mt19937 &gen, double &out_min_dist2)
{
    const std::size_t n = points.size();
    const std::size_t dim = params.dimension;
    const double target_dist2 = 1.0;
    const double eps = 1e-12;

    double best_min_dist2 = 0.0;

    for (std::size_t iter = 0; iter < params.max_iters; ++iter)
    {
        std::vector<Point> forces(n, Point(dim, 0.0));
        double min_dist2 = std::numeric_limits<double>::infinity();

        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t j = i + 1; j < n; ++j)
            {
                Point diff(dim);
                double dist2 = 0.0;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    double delta = points[i][d] - points[j][d];
                    diff[d] = delta;
                    dist2 += delta * delta;
                }

                min_dist2 = std::min(min_dist2, dist2);
                double dist = std::sqrt(dist2) + eps;

                if (dist2 < target_dist2)
                {
                    double gap = target_dist2 - dist2;
                    double scale = (params.penalty_weight * gap) / dist;
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        double push = diff[d] * scale;
                        forces[i][d] += push;
                        forces[j][d] -= push;
                    }
                }

                if (params.repulsion > 0.0)
                {
                    double rep = params.repulsion / (dist2 * dist + eps);
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        double push = diff[d] * rep;
                        forces[i][d] += push;
                        forces[j][d] -= push;
                    }
                }
            }
        }

        double t = static_cast<double>(iter) / static_cast<double>(params.max_iters);
        double step = params.step_end + (params.step_start - params.step_end) * (1.0 - t);
        step /= std::sqrt(static_cast<double>(n));

        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                points[i][d] += step * forces[i][d];
            }
            normalize(points[i]);
        }

        best_min_dist2 = std::max(best_min_dist2, min_dist2);

        if (min_dist2 >= target_dist2 - params.tolerance)
        {
            out_min_dist2 = min_dist2;
            return true;
        }

        if (params.jitter > 0.0 && params.jitter_every > 0 && (iter + 1) % params.jitter_every == 0)
        {
            std::normal_distribution<double> noise(0.0, params.jitter * (1.0 - t));
            for (std::size_t i = 0; i < n; ++i)
            {
                for (std::size_t d = 0; d < dim; ++d)
                {
                    points[i][d] += noise(gen);
                }
                normalize(points[i]);
            }
        }
    }

    out_min_dist2 = best_min_dist2;
    return false;
}

static std::string format_points(const Points &points)
{
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(3);
    oss << "[";
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        if (i > 0)
        {
            oss << ", ";
        }
        oss << "(";
        for (std::size_t d = 0; d < points[i].size(); ++d)
        {
            if (d > 0)
            {
                oss << ", ";
            }
            oss << points[i][d];
        }
        oss << ")";
    }
    oss << "]";
    return oss.str();
}

static void run_dimension(std::size_t dim,
                          std::size_t max_points,
                          const SearchParams &base_params,
                          const Points &seed_points,
                          const std::string &seed_label)
{
    SearchParams params = base_params;
    params.dimension = dim;

    std::random_device rd;
    std::mt19937 gen(rd());

    Points best_points = seed_points;
    double best_min_dist2 = 0.0;

    if (!best_points.empty())
    {
        best_min_dist2 = min_distance_squared(best_points);
        std::cout << "d=" << dim
                  << " seed=" << seed_label
                  << " N=" << best_points.size()
                  << " min_dist=" << std::sqrt(best_min_dist2)
                  << std::endl;
    }

    if (best_points.size() >= max_points && !best_points.empty())
    {
        std::cout << "d=" << dim << " result K=" << best_points.size()
                  << " min_dist=" << std::sqrt(best_min_dist2) << std::endl;
        if (best_points.size() <= 60)
        {
            std::cout << format_points(best_points) << std::endl;
        }
        else
        {
            std::cout << "point list omitted (N > 60)" << std::endl;
        }
        return;
    }

    std::size_t start_n = best_points.empty() ? 2 : best_points.size() + 1;
    for (std::size_t n = start_n; n <= max_points; ++n)
    {
        bool success = false;
        double best_attempt_min_dist2 = 0.0;

        if (!best_points.empty())
        {
            for (std::size_t attempt_idx = 0; attempt_idx < params.insert_restarts && !success; ++attempt_idx)
            {
                Points attempt = insert_point_best(best_points, params.insert_candidates, gen);
                if (attempt.size() != n)
                {
                    continue;
                }
                double min_dist2 = 0.0;
                if (optimize_points(attempt, params, gen, min_dist2))
                {
                    success = true;
                    best_points = attempt;
                    best_min_dist2 = min_dist2;
                }
                else
                {
                    best_attempt_min_dist2 = std::max(best_attempt_min_dist2, min_dist2);
                }
            }
        }

        for (std::size_t restart = 0; !success && restart < params.restarts; ++restart)
        {
            Points attempt = random_points(n, dim, gen);
            double min_dist2 = 0.0;
            if (optimize_points(attempt, params, gen, min_dist2))
            {
                success = true;
                best_points = attempt;
                best_min_dist2 = min_dist2;
                break;
            }
            best_attempt_min_dist2 = std::max(best_attempt_min_dist2, min_dist2);
        }

        if (success)
        {
            std::cout << "d=" << dim
                      << " N=" << n
                      << " success min_dist=" << std::sqrt(best_min_dist2)
                      << std::endl;
        }
        else
        {
            std::cout << "d=" << dim
                      << " N=" << n
                      << " failed best_min_dist=" << std::sqrt(best_attempt_min_dist2)
                      << std::endl;
            break;
        }
    }

    if (!best_points.empty())
    {
        std::cout << "d=" << dim << " result K=" << best_points.size()
                  << " min_dist=" << std::sqrt(best_min_dist2) << std::endl;
        if (best_points.size() <= 60)
        {
            std::cout << format_points(best_points) << std::endl;
        }
        else
        {
            std::cout << "point list omitted (N > 60)" << std::endl;
        }
    }
}

static SearchParams default_params_for_dimension(std::size_t dim)
{
    SearchParams params;
    if (dim == 2)
    {
        params.max_iters = 4000;
        params.restarts = 30;
        params.insert_candidates = 120;
        params.insert_restarts = 20;
        params.step_start = 0.05;
        params.step_end = 0.005;
        params.repulsion = 0.01;
        params.penalty_weight = 2.0;
        params.jitter = 0.005;
        params.jitter_every = 200;
        params.tolerance = 1e-4;
    }
    else if (dim == 3)
    {
        params.max_iters = 20000;
        params.restarts = 120;
        params.insert_candidates = 600;
        params.insert_restarts = 30;
        params.step_start = 0.08;
        params.step_end = 0.005;
        params.repulsion = 0.02;
        params.penalty_weight = 4.0;
        params.jitter = 0.01;
        params.jitter_every = 200;
        params.tolerance = 1e-6;
    }
    else if (dim == 4)
    {
        params.max_iters = 6000;
        params.restarts = 20;
        params.insert_candidates = 200;
        params.insert_restarts = 20;
        params.step_start = 0.05;
        params.step_end = 0.004;
        params.repulsion = 0.01;
        params.penalty_weight = 2.5;
        params.jitter = 0.006;
        params.jitter_every = 200;
        params.tolerance = 1e-6;
    }
    else if (dim == 8)
    {
        params.max_iters = 4000;
        params.restarts = 0;
        params.insert_candidates = 400;
        params.insert_restarts = 10;
        params.step_start = 0.05;
        params.step_end = 0.003;
        params.repulsion = 0.01;
        params.penalty_weight = 2.0;
        params.jitter = 0.004;
        params.jitter_every = 200;
        params.tolerance = 1e-6;
    }
    else if (dim == 13)
    {
        params.max_iters = 5000;
        params.restarts = 0;
        params.insert_candidates = 600;
        params.insert_restarts = 8;
        params.step_start = 0.04;
        params.step_end = 0.002;
        params.repulsion = 0.01;
        params.penalty_weight = 3.0;
        params.jitter = 0.003;
        params.jitter_every = 250;
        params.tolerance = 1e-6;
    }
    else
    {
        params.max_iters = 5000;
        params.restarts = 20;
        params.insert_candidates = 200;
        params.insert_restarts = 10;
        params.step_start = 0.05;
        params.step_end = 0.003;
        params.repulsion = 0.01;
        params.penalty_weight = 2.0;
        params.jitter = 0.005;
        params.jitter_every = 200;
        params.tolerance = 1e-6;
    }
    return params;
}

static std::size_t default_target_for_dimension(std::size_t dim)
{
    if (dim == 2)
    {
        return 10;
    }
    if (dim == 3)
    {
        return 15;
    }
    if (dim == 4)
    {
        return 30;
    }
    if (dim == 8)
    {
        return 245;
    }
    if (dim == 13)
    {
        return 593;
    }
    return dim * 2;
}

int main(int argc, char **argv)
{
    if (argc > 1 && std::string(argv[1]) == "--dim")
    {
        if (argc < 3)
        {
            std::cerr << "Usage: " << argv[0] << " --dim <D> [target]" << std::endl;
            return 1;
        }
        std::size_t dim = static_cast<std::size_t>(std::stoul(argv[2]));
        std::size_t target = (argc > 3) ? static_cast<std::size_t>(std::stoul(argv[3]))
                                        : default_target_for_dimension(dim);

        std::string seed_label;
        Points seed_points = seed_points_for_dimension(dim, seed_label);
        SearchParams params = default_params_for_dimension(dim);

        run_dimension(dim, target, params, seed_points, seed_label);
        return 0;
    }

    if (argc > 1)
    {
        std::string seed_path = argv[1];
        std::size_t target = 0;
        if (argc > 2)
        {
            target = static_cast<std::size_t>(std::stoul(argv[2]));
        }

        Points seed_points;
        std::size_t dim = 0;
        if (!load_points_from_file(seed_path, seed_points, dim))
        {
            std::cerr << "Failed to load seed file: " << seed_path << std::endl;
            return 1;
        }
        if (target == 0)
        {
            target = seed_points.size();
        }

        SearchParams params = default_params_for_dimension(dim);

        run_dimension(dim, target, params, seed_points, "file");
        return 0;
    }

    SearchParams params2 = default_params_for_dimension(2);
    run_dimension(2, default_target_for_dimension(2), params2, {}, "");

    SearchParams params3 = default_params_for_dimension(3);
    run_dimension(3, default_target_for_dimension(3), params3, {}, "");

    std::string seed_label;
    Points seed4 = seed_points_for_dimension(4, seed_label);
    SearchParams params4 = default_params_for_dimension(4);
    run_dimension(4, default_target_for_dimension(4), params4, seed4, seed_label);

    seed_label.clear();
    Points seed8 = seed_points_for_dimension(8, seed_label);
    SearchParams params8 = default_params_for_dimension(8);
    run_dimension(8, default_target_for_dimension(8), params8, seed8, seed_label);

    return 0;
}
