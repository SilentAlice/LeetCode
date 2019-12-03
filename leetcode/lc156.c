#include "leetcode.h"

/* 给定一个二叉树，其中所有的右节点要么是具有兄弟节点（拥有相同父节点的左节点）的叶节点，要么为空，将此二叉树上下翻转并将它变成一棵树， 原来的右节点将转换成左叶节点。返回新的根。

例子:

输入: [1,2,3,4,5]

    1
   / \
  2   3
 / \
4   5

输出: 返回二叉树的根 [4,5,2,#,#,3,1]

   4
  / \
 5   2
    / \
   3   1  

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/binary-tree-upside-down
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

/**
 * Definition for a binary tree node.
 */
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
};

struct TreeNode *upsideDownBinaryTree(struct TreeNode *root)
{
    if (root == NULL || root->left == NULL)
        return root;

    struct TreeNode *left = root->left;
    struct TreeNode *right = root->right;
    struct TreeNode *ret;
    root->left = NULL;
    root->right = NULL;

    ret = upsideDownBinaryTree(left);
    left->left = right;
    left->right = root;
    return ret;
}

int main(void)
{
#define CASE_NR 5
    struct TreeNode root[CASE_NR] = {
        {.val = 0, .left = &root[1], .right = &root[2]},
        {.val = 1, .left = &root[3], .right = &root[4]},
        {.val = 2, .left = NULL, .right = NULL},
        {.val = 3, .left = NULL, .right = NULL},
        {.val = 4, .left = NULL, .right = NULL}
    };
    struct TreeNode *ret;

    ret = upsideDownBinaryTree(root);
    for (int i = 0; i < CASE_NR; i++) {
        printf("%d ", root[i].val);
    }
    printf("\n");
    return 0;
}