#include <iostream>
#include <queue>
#include <omp.h> // Include OpenMP for parallel processing

using namespace std;

class Node
{
public:
    int data;
    Node *left, *right;

    Node(int val)
    {
        data = val;
        left = right = nullptr;
    }
};

class BreadthFS
{
public:
    Node *insert(Node *, int);
    void bfs(Node *);
};

// Function to insert a node in a binary tree (level order insertion)
Node *BreadthFS::insert(Node *root, int data)
{
    if (!root)
    {
        return new Node(data);
    }

    queue<Node *> q;
    q.push(root);

    while (!q.empty())
    {
        Node *temp = q.front();
        q.pop();

        if (temp->left == nullptr)
        {
            temp->left = new Node(data);
            return root;
        }
        else
        {
            q.push(temp->left);
        }

        if (temp->right == nullptr)
        {
            temp->right = new Node(data);
            return root;
        }
        else
        {
            q.push(temp->right);
        }
    }
    return root;
}

// Function to perform Breadth-First Search (BFS) traversal
void BreadthFS::bfs(Node *head)
{
    if (!head)
        return;

    queue<Node *> q;
    q.push(head);

    while (!q.empty())
    {
        int qSize = q.size();

#pragma omp parallel for // Parallel processing
        for (int i = 0; i < qSize; i++)
        {
            Node *currNode;

#pragma omp critical // Ensure thread-safe access
            {
                currNode = q.front();
                q.pop();
                cout << "\t" << currNode->data;
            }

#pragma omp critical // Thread-safe queue operations
            {
                if (currNode->left)
                    q.push(currNode->left);
                if (currNode->right)
                    q.push(currNode->right);
            }
        }
    }
}

int main()
{
    BreadthFS tree;
    Node *root = nullptr;
    int data;
    char ans;

    do
    {
        cout << "\nEnter data: ";
        cin >> data;
        root = tree.insert(root, data);

        cout << "Do you want to insert another node? (y/n): ";
        cin >> ans;
    } while (ans == 'y' || ans == 'Y');

    cout << "\nBreadth-First Search (BFS) traversal:";
    tree.bfs(root);

    return 0;
}