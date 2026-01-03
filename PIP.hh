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
        auto joined = coords | std::views::transform([](double c)
                                                     { return std::format("{:.3f}", c); }) |
                      std::views::join_with(std::string_view(", "));
        return std::format("({})", std::ranges::to<std::string>(joined));
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
                if (points[i].squareDistanceTo(points[j]) <= 1) // 我是大笨蛋，一开始竟然写的4
                    return false;
        return true;
    }

    double fitness() const // 计算个体适应度
    {
        if (!isValid())
            return 0.0; // 非法解被淘汰
        double fit = 0.0;
        std::size_t sz = points.size();
        for (std::size_t i = 0; i < sz; ++i)
            for (std::size_t j = i + 1; j < sz; ++j)
                fit += points[i].squareDistanceTo(points[j]);
        return fit / sz + sz * 1000.0; // 鼓励更多的点，同时考虑点之间的平均距离越大越好
    }

    void mutate(double mutationRate) // 个体变异
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        if (points.empty())
            return;

        std::uniform_int_distribution<> pointDis(0, points.size() - 1);

        // 随机替换一些点
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            if (dis(gen) < mutationRate) // 以 mutationRate 的概率替换点
            {
                points[i] = Point<N>();
            }
        }

        // 尝试添加新点
        // 由于本项目的目标是尽可能多地放置点，因而我们极为鼓励添加新点的变异操作，所以这里的概率直接倍增
        if (dis(gen) < mutationRate * 2.0)
        {
            points.push_back(Point<N>());
        }

        // // 尝试删除点，实际上不需要，因为我们是从下往上计算的，所以点越多越好，肯定不允许删除点
        // if (points.size() > 1 && dis(gen) < mutationRate)
        // {
        //     std::size_t idxToRemove = pointDis(gen);
        //     points.erase(points.begin() + idxToRemove);
        // }
    }

    std::string toString() const // 转成字符串的方法，便于打印
    {
        auto joined = points | std::views::transform([](const Point<N> &p)
                                                     { return p.toString(); }) |
                      std::views::join_with(std::string_view(", "));
        return std::format("[{}]", std::ranges::to<std::string>(joined));
    }
};

template <std::size_t N>
class Population // 用于遗传算法的“种群”，包含多个个体
{
private:
    std::vector<Individual<N>> individuals;

public:
    Population(std::size_t populationSize, std::size_t pointCount = 2 * N) // 初始化种群，默认每个个体包含2*N个点
    {
        // 先清空并预留空间
        individuals.clear();
        individuals.reserve(populationSize);

        // 我突然反应过来：初始数据为什么真要随机生成呢？不如直接生成一个合法解，把它复制n次，剩下的交给变异和交叉！
        // 这样的合法解可以用一个简单的规则来生成，比如d维度的情况下，放置2*N个点，分别在每个维度的+1和-1位置上
        // 而且这些点之间的最小距离都至少是sqrt(2)>1，完全合法！
        std::vector<Point<N>> initialPoints;
        for (std::size_t dim = 0; dim < N; ++dim) // 按照上述逻辑生成初始点，并用这些点构造个体
        {
            if (dim / 2 > pointCount)
                break; // 如果点数已经够了就停止
            std::array<double, N> posPlus = {};
            std::array<double, N> posMinus = {};
            posPlus[dim] = 1.0;
            posMinus[dim] = -1.0;
            initialPoints.emplace_back(posPlus);
            initialPoints.emplace_back(posMinus);
        }

        for (std::size_t i = 0; i < populationSize; ++i) // 复制若干次
        {
            individuals.emplace_back(initialPoints);
        }
    }

    const Individual<N> &getBestIndividual() const // 找到适应度最高的个体，相同返回第一个（其实是可以的）
    {
        return *std::max_element(
            individuals.begin(), individuals.end(),
            [](const Individual<N> &a, const Individual<N> &b)
            {
                return a.fitness() < b.fitness();
            });
    }

    void evolve(std::size_t generations, double mutationRate) // 进化若干代
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (std::size_t genIdx = 0; genIdx < generations; ++genIdx)
        {
            // 评估适应度
            std::vector<std::pair<double, Individual<N>>> fitnessIndividuals;
            for (auto &ind : individuals)
            {
                fitnessIndividuals.emplace_back(ind.fitness(), ind);
            }

            // 按适应度排序
            std::sort(fitnessIndividuals.begin(), fitnessIndividuals.end(),
                      [](const auto &a, const auto &b)
                      { return a.first > b.first; });

            // 选择前半部分个体作为父代
            std::vector<Individual<N>> newGeneration;
            for (std::size_t i = 0; i < fitnessIndividuals.size() / 2; ++i)
            {
                newGeneration.push_back(fitnessIndividuals[i].second);
            }

            // 交叉生成新个体
            std::uniform_int_distribution<> parentDis(0, newGeneration.size() - 1);
            while (newGeneration.size() < individuals.size())
            {
                const Individual<N> &parent1 = newGeneration[parentDis(gen)];
                const Individual<N> &parent2 = newGeneration[parentDis(gen)];

                // 交叉：取两个父代的点混合
                const auto &points1 = parent1.getPoints();
                const auto &points2 = parent2.getPoints();

                std::vector<Point<N>> childPoints;

                // 取 parent1 的一半点
                std::size_t half1 = points1.size() / 2;
                childPoints.insert(childPoints.end(),
                                   points1.begin(),
                                   points1.begin() + half1);

                // 取 parent2 的后一半点
                std::size_t half2 = points2.size() / 2;
                childPoints.insert(childPoints.end(),
                                   points2.begin() + half2,
                                   points2.end());
                // 实际上这也挺粗糙的，感觉不如随机交叉，但这么写肯定够快（笑）

                // 使用新的构造函数创建子代
                Individual<N> child(childPoints);

                newGeneration.push_back(child);
            }

            // 变异
            for (auto &ind : newGeneration)
            {
                ind.mutate(mutationRate);
            }

            // 确保上一代最佳个体被保留，这一个非常重要，确保进化肯定不会退步
            newGeneration[0] = getBestIndividual();

            // 因为点越多越好，所以子代中点的数量比亲代少的那部分被上一代最佳个体替换掉
            // 当然这是一个非常粗糙且偷懒的做法，实际上可以设计得更好，例如随机插入点，直到合法为止
            std::size_t minPointCount = getBestIndividual().getPointCount();
            for (auto &ind : newGeneration)
            {
                while (ind.getPointCount() < minPointCount)
                {
                    ind.getPoints().emplace_back(Point<N>()); // 我们这里采取了直接添加随机点的做法
                }
            }

            individuals = std::move(newGeneration);
        }
    }
};