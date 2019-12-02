#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stddef.h>
#include <assert.h>
#include <ctype.h>
#include <math.h>

/* Array */
#define LEN 10
#define ARRAY_SIZE(n) (sizeof(n) / sizeof(n[0]))

struct Node {
    int val;
    struct Node *next;
};

struct StaticNode {
    int val;
    int next;
};

static int array[LEN] = {0, 1, 2, 3};
static struct Node list[LEN] = {
    {.val = 0, .next = &list[1]},
    {.val = 1, .next = &list[2]},
    {.val = 2, .next = &list[3]},
    {.val = 3, .next = NULL}
    };
static struct StaticNode staticList[LEN] = {
    {.val = -1,.next = 1},  /* dummy node */
    {.val = 0, .next = 2},
    {.val = 1, .next = 3},
    {.val = 2, .next = 4},
    {.val = 3, .next = -1}
};

int main(void) {
    /* array */
    for (int i = 0; i < ARRAY_SIZE(array); i++) {
        printf("%d ", array[i]);
    }
    printf("\n");

    /* list */
    struct Node *pList = list;
    while (pList != NULL) {
        printf("%d ", pList->val);
        pList = pList->next;
    }
    printf("\n");

    /* static list */
    struct StaticNode *pslist = staticList;
    do {
        pslist = &staticList[pslist->next];
        printf("%d ", pslist->val);
    } while (pslist->next >= 0);
    printf("\n");

    return 0;
}