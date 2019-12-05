#include "leetcode.h"

/* 给定一个整型数组，在数组中找出由三个数组成的最大乘积，并输出这个乘积。

示例 1:

输入: [1,2,3]
输出: 6
示例 2:

输入: [1,2,3,4]
输出: 24
注意:

给定的整型数组长度范围是[3,104]，数组中所有的元素范围是[-1000, 1000]。
输入的数组中任意三个数的乘积不会超出32位有符号整数的范围。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/maximum-product-of-three-numbers
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

int compare(const void *a, const void *b)
{
    return *(int *)a - *(int *)b;
}

int maximumProduct(int *nums, int numsSize)
{
    int p1, p2;
    qsort(nums, numsSize, sizeof(int), compare);
    p1 = nums[0] * nums[1] * nums[numsSize - 1];
    p2 = nums[numsSize - 1] * nums[numsSize - 2] * nums[numsSize - 3];
    return p1 > p2 ? p1 : p2;
}

int main(void)
{
#define CASE_NR 4
    int input[CASE_NR] = {1, 2, 3, 4};
    int ret = maximumProduct(input, CASE_NR);
    printf("%d\n", ret);
    return 0;
}

