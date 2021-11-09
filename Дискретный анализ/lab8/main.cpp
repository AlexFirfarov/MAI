#include <iostream>
#include <vector>
#include <queue>

const int ERROR = -1;

int main() {
    int n = 0;
    int m = 0;
    int u;
    int v;

    std::cin >> n >> m;

    std::vector<std::vector<int>> top(n);
    std::vector<int> inDegree(n);
    std::vector<int> result(n);
    int count = 0;

    for (int i = 0; i < m; ++i) {
        std::cin >> u >> v;
        top[u - 1].push_back(v - 1);
        ++inDegree[v - 1];
    }

    std::queue<int> q;
    for (int i = 0; i < n; ++i) {
        if (!inDegree[i]) {
            q.push(i);
            ++count;
        }
    }
    int pos = 0;
    while (!q.empty()) {
        int temp = q.front();
        q.pop();
        result[pos] = temp + 1;
        ++pos;
        for (int v = 0; v < top[temp].size(); ++v) {
            --inDegree[top[temp][v]];
            if (inDegree[top[temp][v]] == 0) {
                q.push(top[temp][v]);
                ++count;
            }
        }
    }
    if (count == n) {
        for (int i = 0; i < n - 1; ++i) {
            std::cout << result[i] << ' ';
        }
        std::cout << result[n - 1];
    }
    else {
        std::cout << ERROR;
    }
    return 0;
}