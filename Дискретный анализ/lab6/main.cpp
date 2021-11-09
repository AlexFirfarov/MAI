#include "BigInt.h"

int main() {

    std::string strNum1, strNum2;
    char op;
    while (std::cin >> strNum1 >> strNum2 >> op) {
        TBigInt num1(strNum1);
        TBigInt num2(strNum2);
        switch(op) {
            case '+': {
                std::cout << num1 + num2 << std::endl;
                break;
            }
            case '-': {
                if (num1 < num2) {
                    std::cout << "Error" << std::endl;
                } 
                else {
                    std::cout << num1 - num2 << std::endl;
                }  
                break;
            }
            case '*': {
                std::cout << num1 * num2 << std::endl;
                break;
            }
            case '/': {
                if (num2 == TBigInt(0)) {
                    std::cout << "Error" << std::endl;
                }
                else {
                    std::cout << num1 / num2 << std::endl;
                }
                break;
            }
            case '^': {
                if (num1 == TBigInt(0)) {
                    if (num2 == TBigInt(0)) {
                        std::cout << "Error" << std::endl;
                    }  
                    else {
                        std::cout << "0" << std::endl;
                    }       
                } 
                else if (num1 == TBigInt(1)) {
                    std::cout << "1" << std::endl;
                } 
                else {
                    std::cout << Power(num1, std::stoi(strNum2)) << std::endl;
                }    
                break;
            }
            case '<': {
                num1 < num2 ? (std::cout << "true" << std::endl) : (std::cout << "false" << std::endl);
                break;
            }
            case '>': {
                num1 > num2 ? (std::cout << "true" << std::endl) : (std::cout << "false" << std::endl);
                break;
            }
            case '=': {
                num1 == num2 ? (std::cout << "true" << std::endl) : (std::cout << "false" << std::endl);
                break;
            }
        }
    }
    return 0;
}