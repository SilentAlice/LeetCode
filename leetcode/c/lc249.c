#include "leetcode.h"

/*
给定一个字符串，对该字符串可以进行 “移位” 的操作，也就是将字符串中每个字母都变为其在字母表中后续的字母，比如："abc" -> "bcd"。这样，我们可以持续进行 “移位” 操作，从而生成如下移位序列：

"abc" -> "bcd" -> ... -> "xyz"
给定一个包含仅小写字母字符串的列表，将该列表中所有满足 “移位” 操作规律的组合进行分组并返回。

示例：

输入: ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"]
输出: 
[
  ["abc","bcd","xyz"],
  ["az","ba"],
  ["acef"],
  ["a","z"]
]

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/group-shifted-strings
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

/**
 * Return an array of arrays of size *returnSize.
 * The sizes of the arrays are returned as *returnColumnSizes array.
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 */

#define RET_SIZE    100
unsigned int pattern[RET_SIZE];
int retSize;
int retColSize[RET_SIZE];

int idxOfPattern(char *string)
{
    int strLen = strlen(string);
    char *str = malloc(strLen + 1);
    unsigned int pstr;

    /* process string */
    for (int i = 0; i < strLen; i++) {
        str[i] = string[i] - 'a' + '0';
    }
    str[strLen] = 0;
    pstr = atoi(str) | 0x1U << 31U;
    free(str);

    for (int i = 0; i < retSize; i++) {
        if (pstr == pattern[i])
            return i;
    }

    /* not found */
    pattern[retSize++] = pstr;
    return retSize - 1;
}

char ***groupStrings(char **strings, int stringsSize, int *returnSize, int **returnColumnSizes)
{
    char ***ret = malloc(RET_SIZE * sizeof(char**));
    retSize = 0;
    for (int i = 0; i < RET_SIZE; i++) {
        pattern[i] = 0x1U << 31U;
        retColSize[i] = 0;
        ret[i] = malloc(sizeof(char *) * RET_SIZE);
    }

    for (int i = 0; i < stringsSize; i++) {
        int idx = idxOfPattern(strings[i]);
        ret[idx][retColSize[idx]++] = strings[i];
    }

    *returnSize = retSize;
    *returnColumnSizes = malloc(sizeof(int) * retSize);
    for (int i = 0; i < retSize; i++) {
        (*returnColumnSizes)[i] = retColSize[i];
    }

    return ret;
}

int main(void)
{
#define CASE_NR 8
    char *strings[CASE_NR] = {
        "abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"};
    char ***ret;
    int returnSize;
    int *returnColSize;

    ret = groupStrings(strings, CASE_NR, &returnSize, &returnColSize);

    for (int i = 0; i < RET_SIZE; i++) {
        free(ret[i]);
    }
    free(ret);

    return 0;
}
