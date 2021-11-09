#include <iostream>
#include <vector>

const long long INF = 10000000000; 

struct TEdge {
    long long first;
    long long second;
    long long weight;
};

void BellmanFord(const std::vector<TEdge>& gr, std::vector<long long>& res);

int main() {
    long long n = 0;
    long long m = 0;
    long long start = 0;
    long long finish = 0;

    std::cin >> n >> m >> start >> finish;

    std::vector<TEdge> gr(m);
    std::vector<long long> res(n, INF);
    res[start - 1] = 0;

    for (long long i = 0; i < m; ++i) {
        std::cin >> gr[i].first >> gr[i].second >> gr[i].weight;
    }

    BellmanFord(gr, res);
    if (res[finish - 1] == INF) {
        std::cout << "No solution" << '\n';
    }
    else {
        std::cout << res[finish - 1] << '\n';
    }
    
    return 0;
}

void BellmanFord(const std::vector<TEdge>& gr, std::vector<long long>& res) {
    long long n = res.size();
    long long m = gr.size();

    for (long long i = 1; i < n; ++i) {
        bool check = false;
        for (long long j = 0; j < m; ++j) {
            if (res[gr[j].first - 1] < INF) {
                if (res[gr[j].second - 1] > res[gr[j].first - 1] + gr[j].weight) {
                    res[gr[j].second - 1] = res[gr[j].first - 1] + gr[j].weight;
                    check = true;
                }
            }
        }
        if (!check) {
            return;
        }
    }
    return;
}