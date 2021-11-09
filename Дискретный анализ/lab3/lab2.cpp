#include "lab2.h"

const int DIFF = 32;

int flag = 0;

AVL::AVL() {
    root = nullptr;
}

int AVL::CompStr(char* str1, char* str2) {
    int size1 = strlen(str1);
    int size2 = strlen(str2);
    if (size1 > size2) {
        return 1;
    }
    else if (size1 < size2) {
        return -1;
    }
    return strcmp(str1, str2);
}

int AVL::BalanceFactor(Node* node) {
    return Height(node->left) - Height(node->right);
}

int AVL::Height(Node* node) {
    if (node) {
        return node->height;
    }
    return 0;
}

void AVL::ChangeHeight(Node* node) {
    int leftHeight = Height(node->left);
    int rightHeight = Height(node->right);
    if (leftHeight > rightHeight) {
        node->height = leftHeight + 1;
    }
    else {
        node->height = rightHeight + 1;
    }
}

Node* AVL::RotateLeft(Node* node) {
    Node* right = node->right;
    node->right = right->left;
    right->left = node;
    ChangeHeight(node);
    ChangeHeight(right);
    return right;
}

Node* AVL::RotateRight(Node* node) {
    Node* left = node->left;
    node->left = left->right;
    left->right = node;
    ChangeHeight(node);
    ChangeHeight(left);
    return left;
}

Node* AVL::UpdateTree(Node* node, int & balance) {
    int balanceChild = 0;
    if (balance == -2) {
        if ((balanceChild = BalanceFactor(node->right)) > 0) {
            node->right = RotateRight(node->right);
        }
        return RotateLeft(node);
    }
    if (balance == 2) {
        if ((balanceChild = BalanceFactor(node->left)) < 0) {
            node->left = RotateLeft(node->left);
        }
        return RotateRight(node);
    }
    return node;
}

void AVL::Search(char* word) {
    int length = strlen(word);
    for (int i = 0; i < length; ++i) {
        if (isupper(word[i])) {
            word[i] = word[i] + DIFF;
        }
    }
    Search(word, root);
}

void AVL::Search(char* word, Node* node) {
    if (node == nullptr) {
        printf("NoSuchWord\n");
        return;
    }
    int result = 0;
    if ((result = CompStr(word, node->word)) < 0) {
        Search(word, node->left);
    }
    else if ((result = CompStr(word, node->word)) > 0) {
        Search(word, node->right);
    }
    else {
        printf("OK: %llu\n", node->value);
        return;
    }
}

void AVL::Insert(char* word, unsigned long long & value) {
    int length = strlen(word);
    for (int i = 0; i < length; ++i) {
        if (isupper(word[i])) {
            word[i] = word[i] + DIFF;
        }
    }

    if (root == nullptr) {
        root = new Node(word, value);
        printf("OK\n");
        return;
    }
    root = Insert(word, value, root);
    flag = 0;
}

Node* AVL::Insert(char* word, unsigned long long & value, Node* node) {
    if (node == nullptr) {
        node = new Node(word, value);
        printf("OK\n");
        return node;
    }
    int result = 0;
    if ((result = CompStr(word, node->word)) < 0) {
        node->left = Insert(word, value, node->left);
    }
    else if ((result = CompStr(word, node->word)) > 0) {
        node->right = Insert(word, value, node->right);
    }
    else {
        printf("Exist\n");
        return node;
    }
    
    if (flag == 0) {
        ChangeHeight(node);
        int balance = BalanceFactor(node);
        if (balance == 0) {
            flag = 1;
            return node;
        }
        else if (balance == 1 || balance == -1) {
            return node;
        }
        return UpdateTree(node, balance);
        flag = 1;
    }
    return node;

}

void AVL::Delete(char* word) {
    int length = strlen(word);
    for (int i = 0; i < length; ++i) {
        if (isupper(word[i])) {
            word[i] = word[i] + DIFF;
        }
    }
    root = Delete(word, root);
    flag = 0;
}

Node* AVL::Delete(char* word, Node* node) {
    if (node == nullptr) {
        printf("NoSuchWord\n");
        return nullptr;
    }
    int result = 0;
    if ((result = CompStr(word, node->word)) < 0) {
        node->left = Delete(word, node->left);
    }
    else if ((result = CompStr(word, node->word)) > 0) {
        node->right = Delete(word, node->right);
    }
    else {
        if (node->left == nullptr && node->right == nullptr) {
            delete [] node->word;
            delete node;
            printf("OK\n");
            return nullptr;
        }
        else if (node->left == nullptr) {
            Node* temp = node->right;
            delete [] node->word;
            delete node;
            printf("OK\n");
            return temp;
        }
        else if (node->right == nullptr) {
            Node* temp = node->left;
            delete [] node->word;
            delete node;
            printf("OK\n");
            return temp;
        }
        else {
            Node* max = FindMax(node->left);
            node->value = max->value;
            int size = strlen(max->word);
            delete [] node->word;
            node->word = new char[size + 1];
            strcpy(node->word, max->word);
            node->left = DeleteMax(node->left);
            printf("OK\n");
            int balance = BalanceFactor(node);
            return UpdateTree(node, balance);
        }
    }
        
    if (flag == 0) {
        ChangeHeight(node);
        int balance = BalanceFactor(node);
        if (balance == 1 || balance == -1) {
            flag = 1;
            return node;
        }
        else if (balance == 0) {
            return node;
        }
        return UpdateTree(node, balance);
    }
    return node;
}

Node* AVL::FindMax(Node* node) {
    if (node->right != nullptr) {
        return FindMax(node->right);
    }
    return node;
}

Node* AVL::DeleteMax(Node* node) {
    if (node->right == nullptr) {
        Node* temp = node->left;
        delete [] node->word;
        delete node;
        return temp;
    }
    node->right = DeleteMax(node->right);
    int balance = BalanceFactor(node);
    return UpdateTree(node, balance);
}

void AVL::Save(FILE* file) {
    Save(root, file);
    printf("OK\n");
    return;
}

void AVL::Save(Node* node, FILE* file) {
    if (node == nullptr) {
        return;
    }
    else {
        bool left = 0;
        bool right = 0;

        if (node->left) {
            left = 1;
        }

        if (node->right) {
            right = 1;
        }
        int size = strlen(node->word);

        fwrite(&size, sizeof(int), 1, file);
        fwrite(&(node->height), sizeof(int), 1, file);
        fwrite(node->word, sizeof(char), size, file);
        fwrite(&(node->value), sizeof(unsigned long long), 1, file);
        fwrite(&left, sizeof(bool), 1, file);
        fwrite(&right, sizeof(bool), 1, file);

        if (left) {
            Save(node->left, file);
        }
        if (right) {
            Save(node->right, file);
        }
    }
    return;
}

void AVL::Load(FILE* file) {
    root = LoadTree(file);
    printf("OK\n");
    return;
}

Node* AVL::LoadTree(FILE* file) {
    bool left = 0;
    bool right = 0;
    int height = 0;
    int size = 0;
    char* word;
    unsigned long long value = 0;

    if (fread(&size, sizeof(int), 1, file) > 0) {
        fread(&height, sizeof(int), 1, file);
        word = new char[size + 1];
        fread(word, sizeof(char), size, file);
        fread(&value, sizeof(unsigned long long), 1, file);
        word[size] = '\0';
        Node* node = new Node(word, value);
        node->height = height;
        delete [] word;
        fread(&left, sizeof(bool), 1, file);
        fread(&right, sizeof(bool), 1, file);

        if (left) {
            node->left = LoadTree(file);
        }
        if (right) {
            node->right = LoadTree(file);
        }

        return node;
    }
    else {
        return nullptr;
    }
}

void AVL::Destroy() {
    Destroy(root);
    return;
}

void AVL::Destroy(Node* node) {
    if (node != nullptr) {
        Destroy(node->left);
        Destroy(node->right);
        delete [] node->word;
        delete node;
    }
}