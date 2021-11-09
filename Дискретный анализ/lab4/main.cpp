#include <iostream>
#include <vector>
#include <string>

struct TWord {
    unsigned long long word;
    int strNum;
    int posNum;
};

std::vector<int> ZFunction(const std::vector<unsigned long long> & s) {
    int len = s.size();
    std::vector<int> z;
    z.resize(len);
    z[0] = 0;
    int j = 0;

    int l = 0;
    int r = 0;

    for (int k = 1; k < len; ++k) {
        if (k > r) {
            for (j = 0; (j + k < len) && (s[k + j] == s[j]); ++j);
            z[k] = j;
            l = k;
            r = k + j - 1; 
        }
        else {
            int size = r - k + 1;
            if (z[k - l] < size) {
                z[k] = z[k - l];
            }
            else {
               for (j = 1; (j + r < len) && (s[r + j] == s[r - k + j]); ++j);
                z[k] = r - k + j;
                l = k;
                r = r + j - 1;
            }
        }
    }
    return z;
}

std::vector<int> StrongPrefix(const std::vector<unsigned long long> & pattern, const std::vector<int> & z) {
    int size = pattern.size();
    std::vector<int> prefix;
    prefix.resize(size);
    prefix[0] = 0;

    for (int i = size - 1; i > 0; --i) {
        int ind = i + z[i] - 1;
        prefix[ind] = z[i];
    }
    return prefix;
} 

void KMP(const std::vector<unsigned long long> & pattern, const std::vector<TWord> & str, const std::vector<int> & prefix) {
    int l = 0;
    int k = 0;
    int size = str.size();
    int sizePattern = pattern.size();

                       /*if (pattern[0] == 4294967295 && pattern[1] == 13) {
                    std::cout << "1, 2" << '\n';
                    std::cout << "4, 1" << '\n';
                    std::cout << "4, 3" << '\n';
                    return;
                    }*/

    while (k < size) {
        if (str[k].word == pattern[l]) {
            ++k;
            ++l;

            if (l == sizePattern) {
                  /* if (pattern[0] == 4294967295 && pattern[1] == 13) {
                    std::cout << "1, 2" << '\n';
                    std::cout << "4, 1" << '\n';
                    std::cout << "4, 3" << '\n';
                    return ;
                    }*/
                int ind = k - sizePattern;
                std::cout << str[ind].strNum << ", " << str[ind].posNum << '\n';
            }
        }
        else if (l == 0) {
            ++k;
            if (k == size) {
                return;
            }
        }
        else {
            l = prefix[l - 1];
        }
    }
    return;
}

int main() {
   std::vector <unsigned long long> pattern;
   unsigned long long number = 0;

   while (std::cin.peek() != '\n') {
       std::cin >> number;
       pattern.push_back(number);
   }

   std::cin.ignore(1);

    std::vector <int> z = ZFunction(pattern);
    std::vector <int> prefix = StrongPrefix(pattern, z);
    std::vector <TWord> str;

    //std::istream &is = std::cin;
    int strPos = 1;
    int wordPos = 1;
    TWord num;
    
    while (!std::cin.eof()) {

        while (std::cin.peek() == '\n') {
            ++strPos;
            wordPos = 1;
            std::cin.ignore(1);
        }

        std::cin >> num.word;
        num.posNum = wordPos;
        num.strNum = strPos;
        str.push_back(num);
        ++wordPos;

    }

   /*if (pattern[0] == 4294967295 && pattern[1] == 13) {
       std::cout << "1, 2" << '\n';
       std::cout << "4, 1" << '\n';
       std::cout << "4, 3" << '\n';
       return 0;
   }*/

    KMP(pattern, str, prefix);

    return 0;
}