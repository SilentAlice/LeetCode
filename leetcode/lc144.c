#include "leetcode.h"

/*
Given a binary tree, return the preorder traversal of its nodes' values.

Example:

Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [1,2,3]
Follow up: Recursive solution is trivial, could you do it iteratively?
*/

/**
 * Definition for a binary tree node.
 */
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
};

/**
 * Note: The returned array must be malloced, assume caller calls free().
 */

#define LIST_SIZE   1000
static int *retArray;
static int idx;

void preOrder(struct TreeNode *root) {
    retArray[idx++] = root->val;
    if (root->left != NULL)
        preOrder(root->left);
    if (root->right != NULL)
        preOrder(root->right);
    return ;
}

int *preorderTraversal(struct TreeNode *root, int *returnSize)
{
    if (root == NULL) {
        *returnSize = 0;
        return NULL;
    }

    retArray = malloc(sizeof(int) * LIST_SIZE);
    idx = 0;

    preOrder(root);
    *returnSize = idx;
    return retArray;
}

int main(void)
{
    int returnSize;
    int *ret;

    struct TreeNode root[3] = {
        {.val = 1, .left = NULL, .right = &root[1]},
        {.val = 2, .left = &root[2], .right = NULL},
        {.val = 3, .left = NULL, .right = NULL}
    };

    ret = preorderTraversal(root, &returnSize);
    free(ret);

    return 0;
}
