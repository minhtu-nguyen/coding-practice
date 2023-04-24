class ListNode:
  def __init__(self, x):
    self.val = x
    self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
      prev, curr = None, head

      while curr:
        temp = curr.next
        curr.next = prev
        prev = curr
        curr = temp
      return prev
"""
Iterative: 2 pointers curr and prev (null) O(n) O(1)
Recursive: O(n) O(n)
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
      if not head:
        return None

      newHead = head
      if head.next:
        newHead = self.reverseList(head.next)
        head.next.next = head
      head.next = None

      return newHead
"""
#Grokking
def reverse(head):
    if not head or not head.next:
        return head
        
    list_to_do = head.next
    reversed_list = head
    reversed_list.next = None
    
    while list_to_do:
        temp = list_to_do
        list_to_do = list_to_do.next
        temp.next = reversed_list
        reversed_list = temp

    return reversed_list