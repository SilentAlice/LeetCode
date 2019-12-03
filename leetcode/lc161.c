#include "leetcode.h"
/* 给定两个字符串 s 和 t，判断他们的编辑距离是否为 1。

注意：

满足编辑距离等于 1 有三种可能的情形：

往 s 中插入一个字符得到 t
从 s 中删除一个字符得到 t
在 s 中替换一个字符得到 t
示例 1：

输入: s = "ab", t = "acb"
输出: true
解释: 可以将 'c' 插入字符串 s 来得到 t。
示例 2:

输入: s = "cab", t = "ad"
输出: false
解释: 无法通过 1 步操作使 s 变为 t。
示例 3:

输入: s = "1203", t = "1213"
输出: true
解释: 可以将字符串 s 中的 '0' 替换为 '1' 来得到 t。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/one-edit-distance
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

typedef int bool;
#define false 0
#define true 1

bool isOneEditDistance(char *s, char *t)
{
    char *s1 = s;
    char *s2 = t;
	int lens = 0;
    int lent = 0;

    if (s != NULL)
        lens = strlen(s);
	if (t != NULL)
        lent = strlen(t);

    if (abs(lens - lent) > 1)
        return false;

    if (strcmp(s, t) == 0)
        return false;

    while (*s1 == *s2) {
        s1++;
        s2++;
    }

	if (lens > lent) {
		s1++;
	} else if (lens < lent) {
		s2++;
	} else {
		s1++;
		s2++;
	}
	
    /* now s1 and s2 should be equal */
    return (strcmp(s1, s2) == 0);
}

int main(void) {
	char s[5] = "abcb";
	char t[4] = "acb";
	bool ret;

	ret = isOneEditDistance(s, t);
	return 0;
}