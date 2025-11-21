#include "leetcode.h"

/* 给 n 个进程，每个进程都有一个独一无二的 PID （进程编号）和它的 PPID （父进程编号）。

每一个进程只有一个父进程，但是每个进程可能会有一个或者多个孩子进程。它们形成的关系就像一个树状结构。只有一个进程的 PPID 是 0 ，意味着这个进程没有父进程。所有的 PID 都会是唯一的正整数。

我们用两个序列来表示这些进程，第一个序列包含所有进程的 PID ，第二个序列包含所有进程对应的 PPID。

现在给定这两个序列和一个 PID 表示你要杀死的进程，函数返回一个 PID 序列，表示因为杀这个进程而导致的所有被杀掉的进程的编号。

当一个进程被杀掉的时候，它所有的孩子进程和后代进程都要被杀掉。

你可以以任意顺序排列返回的 PID 序列。

示例 1:
输入:
pid =  [1, 3, 10, 5]
ppid = [3, 0, 5, 3]
kill = 5
输出: [5,10]
解释:
           3
         /   \
        1     5
             /
            10
杀掉进程 5 ，同时它的后代进程 10 也被杀掉。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/kill-process
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

*/

int compare(const void *a, const void *b)
{
	return *(int *)a - *(int *)b;
}

int * killProcess(int *pid, int pidSize, int *ppid, int ppidSize, int kill, int *returnSize)
{
	/* item[0]: ppid, item[1]: pid */
	int ppids[pidSize][2];
	int *parent;
	int *child;

	for (int i = 0; i < pidSize; i++) {
		ppids[i][0] = ppid[i];
		ppids[i][1] = pid[i];
	}

	qsort(ppids, pidSize, sizeof(int) * 2, compare);
	parent = bsearch(&kill, ppids, pidSize, sizeof(int) * 2, compare);
	if (parent == NULL) {
		return NULL;
	}

	child = (parent + 1)


}


int main(void)
{
#define INPUT_LEN	4
	int pid[INPUT_LEN] = {1, 3, 10, 5};
	int ppid[INPUT_LEN] = {3, 0, 5, 3};
	int kill = 5;
	int *ret;
	int retSize;

	ret = killProcess(pid, INPUT_LEN, ppid, INPUT_LEN, kill, &retSize);
	free(ret);
	return 0;
}
