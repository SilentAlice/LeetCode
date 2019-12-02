#include "leetcode.h"

/*
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example:

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
*/

/**
 * Definition for singly-linked list.
 */
struct ListNode {
    int val;
    struct ListNode *next;
};

struct ListNode *addTwoNumbers(struct ListNode *l1, struct ListNode *l2)
{
    int carry = 0;
    int newVal;
    struct ListNode *ret = malloc(sizeof(struct ListNode));
    ret->next = NULL;

    struct ListNode *p1 = l1;
    struct ListNode *p2 = l2;
    struct ListNode *pr = ret;

    do {
        newVal = p1->val + p2->val + carry;
        if (newVal > 9) {
            carry = 1;
            newVal %= 10;
        } else {
            carry = 0;
        }

        pr->next = malloc(sizeof(struct ListNode));
        pr = pr->next;
        pr->val = newVal;

        p1 = p1->next;
        p2 = p2->next;
    } while (p1 != NULL && p2 != NULL);

    p1 = p1 != NULL ? p1 : p2;
    while (p1 != NULL) {
        newVal = p1->val + carry;
        if (newVal > 9) {
            carry = 1;
            newVal %= 10;
        } else {
            carry = 0;
        }

        pr->next = malloc(sizeof(struct ListNode));
        pr = pr->next;
        pr->val = newVal;

        p1 = p1->next;
    };

    /* carry may still be 1 */
    if (carry != 0) {
        pr->next = malloc(sizeof(struct ListNode));
        pr = pr->next;
        pr->val = 1;
    }

    pr->next = NULL;

    /* finish calculation, free dummy node and return */
    pr = ret->next;
    free(ret);
    ret = pr;
    return ret;
}

int main(void)
{
    struct ListNode *ret;
    struct ListNode num0[3] = {
        {.val = 2, .next = &num0[1]},
        {.val = 4, .next = &num0[2]},
        {.val = 3, .next = NULL},
    };
    struct ListNode num1[3] = {
        {.val = 5, .next = &num1[1]},
        {.val = 6, .next = &num1[2]},
        {.val = 4, .next = NULL},
    };

    ret = addTwoNumbers(num0, num1);

    while (ret != NULL) {
        struct ListNode *p = ret->next;
        free(ret);
        ret = p;
    }
    return 0;
}
