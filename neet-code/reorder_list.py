class Solution:
    def reorderList(self, head: ListNode) -> None:
        # find middle
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # reverse second half
        second = slow.next
        prev = slow.next = None
        while second:
            tmp = second.next
            second.next = prev
            prev = second
            second = tmp

        # merge two halfs
        first, second = head, prev
        while second:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first, second = tmp1, tmp2

"""
Possible using array
Better in-place: Split 2 lists, reverese the 2nd using FS pointers to find middle
"""

#Grokking
def reorder_list(head):
    if not head:
        return head
    
    # find the middle of linked list
    # in 1->2->3->4->5->6 find 4 
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next 
        
    # reverse the second part of the list
    # convert 1->2->3->4->5->6 into 1->2->3 and 6->5->4
    # reverse the second half in-place
    prev, curr = None, slow
    while curr:
        curr.next, prev, curr = prev, curr, curr.next       

    # merge two sorted linked lists
    # merge 1->2->3 and 6->5->4 into 1->6->2->5->3->4
    first, second = head, prev
    while second.next:
        first.next, first = second, first.next
        second.next, second = first, second.next
    
    return head