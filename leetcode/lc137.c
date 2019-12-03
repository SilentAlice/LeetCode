#include "leetcode.h"

/*
给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。
说明：
你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

示例 1:

输入: [2,2,3,2]
输出: 3
示例 2:

输入: [0,1,0,1,0,1,99]
输出: 99

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/single-number-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

#define NUM_BIT 32

static int count[NUM_BIT];

int singleNumber(int* nums, int numsSize){
    int ret = 0;
    for (int i = 0; i < NUM_BIT; i++) {
        count[i] = 0;
    }
    
    for (int i = 0; i < numsSize; i++) {
        /* for each num, we calculate bits on each bit */
        for (int j = 0; j < NUM_BIT; j++) {
            if (nums[i] & (0x1U << j))
                count[j]++;
        }
    }
    
    /* except for the 1 time num, other numbers contribute three bits
     * on each bit position, so we mod 3 to remove them */
    for (int i = 0; i < NUM_BIT; i++) {
        if (count[i] % 3)
            ret |= 0x1U << i;
    }
    
    return ret;
}

int main(void)
{
#define CASE_NR 4
    int caseTest[CASE_NR] = {2, 2, 3, 2};
    int ret;
    ret = singleNumber(caseTest, CASE_NR);
    printf("%d\n", ret);

    return 0;
}

/* if time is allowed, we could use qsort and check i and i + k to find needed num */