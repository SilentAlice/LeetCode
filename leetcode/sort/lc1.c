#include "leetcode.h"

/* Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:

Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].

*/

static int compare(const void *a, const void *b)
{
    return *(int *)a - *(int *)b;
}

int* twoSum(int* nums, int numsSize, int target, int* returnSize){
    int *p1, *p2;
    int *ret;
    int idx1, idx2, sum;
    int idxNums[numsSize][2];

    if (nums == NULL || numsSize <= 0)
        return NULL;
    if (numsSize == 1)
        return NULL;

    assert(idxNums != NULL);
    for (int i = 0; i < numsSize; i++) {
        idxNums[i][0] = nums[i];
        idxNums[i][1] = i;
    }

    qsort(idxNums, numsSize, sizeof(int) * 2, compare);

    idx1 = 0;
    idx2 = numsSize - 1;
    sum = idxNums[idx1][0] + idxNums[idx2][0];
    while (sum != target) {
        if (sum < target)
            idx1 += 1;
        if (sum > target)
            idx2 -= 1;
        sum = idxNums[idx1][0] + idxNums[idx2][0];
    }

    /* sum is equal to target */
    ret = malloc(sizeof(int) * 2);
    assert(ret != NULL);
    ret[0] = idxNums[idx1][1];
    ret[1] = idxNums[idx2][1];
    *returnSize = 2;

    return ret;
}


int main(void)
{
    int nums[10] = {2, 11, 7, 15};
    int target = 9;
    int ret_size;
    int *ret = (twoSum(nums, 4, target, &ret_size));

    free(ret);
    return 0;
}
