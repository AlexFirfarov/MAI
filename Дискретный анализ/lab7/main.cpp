#include <iostream>
#include <string>

int main() {
    std::string n;
    int m;
    long long leftBorder = 1;
    long long rightBorder = 0;
    long long first = 0;
    long long last = 0;
    long long answer = 0;

    std::cin >> n >> m;
    for (int i = 0; i < n.length(); ++i) {
        if (i != 0) {
            leftBorder *= 10;
            rightBorder *= 10;
        }
        rightBorder += n[i] - '0';
        if (leftBorder % m) {
            first = leftBorder - leftBorder % m + m;
        }
        else {
            first = leftBorder;
        }
        last = rightBorder - rightBorder % m;
        if (first <= last) {
            answer += (last - first) / m + 1;
        }
    }
    if (last == rightBorder) {
        --answer;
    }
    std::cout << answer << '\n';
    return 0;
}

