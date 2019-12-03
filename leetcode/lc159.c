#include "leetcode.h"

/* 给定一个字符串 s ，找出 至多 包含两个不同字符的最长子串 t 。

示例 1:
输入: "eceba"
输出: 3
解释: t 是 "ece"，长度为3。
示例 2:
输入: "ccaabbb"
输出: 5
解释: t 是 "aabbb"，长度为5。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/longest-substring-with-at-most-two-distinct-characters
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

#define STR_LEN 10000
#define WORD_NR 256
static char *subStr = NULL;

static int count[WORD_NR];
static int total;
static int retMax;

int lengthOfLongestSubstringTwoDistinct(char * s){
    int strLen;
    char *p1 = s;
    char *p2 = s;
    if (s == NULL)
        return 0;

    strLen = strlen(s);
    if (strLen <= 2)
        return strLen;

    /* initialization */
    total = 0;
    retMax = 0;
    p1 = s;
    p2 = s;
    for (int i = 0; i < WORD_NR; i++)
        count[i] = 0;


    for (int i = 0; i < strLen; i++) {
        int idx = *p2;
        if (count[idx] > 0) {
            /* already used */
			count[idx]++;
            p2++;
            continue;
        }

        /* count == 0, new char */
        if (total < 2) {
            total += 1;
            count[idx]++;
            p2++;
            continue;
        }

        /* a valid substring */
        retMax = retMax > p2 - p1 ? retMax : p2 - p1;
		while (1) {
            int idx2 = *p1++;
            count[idx2]--;

            if (count[idx2] == 0) {
                break;
            }
        }

        /* p1 is cleared */
        count[idx]++;
        p2++;
    }

    /* last substring */
    retMax = retMax > p2 - p1 ? retMax : p2 - p1;
    return retMax;
}

int main(void)
{
	char str[12] = "ababacccccc";
	int ret = lengthOfLongestSubstringTwoDistinct(str);
	printf("%d\n", ret);
	return 0;
}