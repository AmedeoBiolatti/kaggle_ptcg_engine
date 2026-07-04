#pragma once
#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace ptcg {

// Minimal std::vector work-alike with inline storage for the first N elements
// and heap fallback beyond, so the hot GameState containers stay allocation-
// free and contiguous in the common case. Iterators are raw pointers with
// std::vector invalidation rules, EXCEPT that moving a SmallVec whose
// elements are inline moves the elements (pointers into the source dangle).
// No allocator support; growth offers the basic exception guarantee only.
template <typename T, size_t N>
class SmallVec {
 public:
  using value_type = T;
  using size_type = size_t;
  using iterator = T*;
  using const_iterator = const T*;
  using reference = T&;
  using const_reference = const T&;

  SmallVec() = default;
  SmallVec(std::initializer_list<T> il) { assign(il.begin(), il.end()); }
  SmallVec(const SmallVec& o) { assign(o.begin(), o.end()); }
  SmallVec(SmallVec&& o) noexcept { steal_or_move(std::move(o)); }
  // std::vector interop for boundary code (copies; hot paths should take the
  // SmallVec directly or a template parameter instead).
  SmallVec(const std::vector<T>& v) { assign(v.begin(), v.end()); }
  // cross-capacity copies (same element type, different inline N)
  template <size_t M>
  SmallVec(const SmallVec<T, M>& o) { assign(o.begin(), o.end()); }
  ~SmallVec() { destroy_all(); }

  SmallVec& operator=(const SmallVec& o) {
    if (this != &o) assign(o.begin(), o.end());
    return *this;
  }
  SmallVec& operator=(SmallVec&& o) noexcept {
    if (this != &o) {
      destroy_all();
      steal_or_move(std::move(o));
    }
    return *this;
  }
  SmallVec& operator=(std::initializer_list<T> il) {
    assign(il.begin(), il.end());
    return *this;
  }
  SmallVec& operator=(const std::vector<T>& v) {
    assign(v.begin(), v.end());
    return *this;
  }
  template <size_t M>
  SmallVec& operator=(const SmallVec<T, M>& o) {
    assign(o.begin(), o.end());
    return *this;
  }
  operator std::vector<T>() const { return std::vector<T>(begin(), end()); }

  template <typename It>
  void assign(It first, It last) {
    clear();
    for (; first != last; ++first) push_back(*first);
  }
  void assign(size_type n, const T& v) {
    clear();
    reserve(n);
    for (size_type i = 0; i < n; ++i) push_back(v);
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  iterator begin() { return data_; }
  const_iterator begin() const { return data_; }
  const_iterator cbegin() const { return data_; }
  iterator end() { return data_ + size_; }
  const_iterator end() const { return data_ + size_; }
  const_iterator cend() const { return data_ + size_; }

  size_type size() const { return size_; }
  bool empty() const { return size_ == 0; }
  size_type capacity() const { return cap_; }

  reference operator[](size_type i) { return data_[i]; }
  const_reference operator[](size_type i) const { return data_[i]; }
  reference front() { return data_[0]; }
  const_reference front() const { return data_[0]; }
  reference back() { return data_[size_ - 1]; }
  const_reference back() const { return data_[size_ - 1]; }

  void reserve(size_type n) {
    if (n > cap_) grow_to(n);
  }

  void clear() {
    for (size_type i = 0; i < size_; ++i) data_[i].~T();
    size_ = 0;
  }

  void push_back(const T& v) {
    if (size_ == cap_) {
      T tmp(v);  // v may alias an element about to be relocated
      grow_to(next_cap());
      ::new (static_cast<void*>(data_ + size_)) T(std::move(tmp));
    } else {
      ::new (static_cast<void*>(data_ + size_)) T(v);
    }
    ++size_;
  }
  void push_back(T&& v) {
    if (size_ == cap_) {
      T tmp(std::move(v));
      grow_to(next_cap());
      ::new (static_cast<void*>(data_ + size_)) T(std::move(tmp));
    } else {
      ::new (static_cast<void*>(data_ + size_)) T(std::move(v));
    }
    ++size_;
  }
  template <typename... Args>
  reference emplace_back(Args&&... args) {
    if (size_ == cap_) grow_to(next_cap());
    ::new (static_cast<void*>(data_ + size_)) T(std::forward<Args>(args)...);
    return data_[size_++];
  }

  void pop_back() {
    data_[--size_].~T();
  }

  void resize(size_type n) {
    while (size_ > n) pop_back();
    reserve(n);
    while (size_ < n) emplace_back();
  }
  void resize(size_type n, const T& v) {
    while (size_ > n) pop_back();
    reserve(n);
    while (size_ < n) push_back(v);
  }

  iterator erase(const_iterator pos) { return erase(pos, pos + 1); }
  iterator erase(const_iterator first, const_iterator last) {
    iterator f = data_ + (first - data_);
    iterator l = data_ + (last - data_);
    iterator new_end = std::move(l, end(), f);
    while (end() != new_end) pop_back();
    return f;
  }

  iterator insert(const_iterator pos, const T& v) {
    T tmp(v);  // v may alias an element of *this
    size_type idx = static_cast<size_type>(pos - data_);
    emplace_back();  // grow by one (may reallocate)
    std::move_backward(data_ + idx, data_ + size_ - 1, data_ + size_);
    data_[idx] = std::move(tmp);
    return data_ + idx;
  }
  iterator insert(const_iterator pos, size_type n, const T& v) {
    T tmp(v);  // v may alias an element of *this
    size_type idx = static_cast<size_type>(pos - data_);
    if (n == 0) return data_ + idx;
    reserve(size_ + n);
    for (size_type i = 0; i < n; ++i) emplace_back();
    std::move_backward(data_ + idx, data_ + size_ - n, data_ + size_);
    for (size_type i = 0; i < n; ++i) data_[idx + i] = tmp;
    return data_ + idx;
  }
  template <typename It>
  iterator insert(const_iterator pos, It first, It last) {
    size_type idx = static_cast<size_type>(pos - data_);
    size_type count = 0;
    for (It it = first; it != last; ++it) ++count;
    if (count == 0) return data_ + idx;
    reserve(size_ + count);
    for (size_type i = 0; i < count; ++i) emplace_back();
    std::move_backward(data_ + idx, data_ + size_ - count, data_ + size_);
    iterator out = data_ + idx;
    for (It it = first; it != last; ++it, ++out) *out = *it;
    return data_ + idx;
  }

  friend bool operator==(const SmallVec& a, const SmallVec& b) {
    return a.size_ == b.size_ && std::equal(a.begin(), a.end(), b.begin());
  }
  friend bool operator!=(const SmallVec& a, const SmallVec& b) {
    return !(a == b);
  }

 private:
  size_type next_cap() const { return cap_ + cap_; }

  bool on_heap() const { return data_ != inline_ptr(); }
  T* inline_ptr() { return reinterpret_cast<T*>(inline_storage_); }
  const T* inline_ptr() const {
    return reinterpret_cast<const T*>(inline_storage_);
  }

  void grow_to(size_type n) {
    size_type cap = cap_;
    while (cap < n) cap += cap;
    T* mem = std::allocator<T>().allocate(cap);
    for (size_type i = 0; i < size_; ++i) {
      ::new (static_cast<void*>(mem + i)) T(std::move(data_[i]));
      data_[i].~T();
    }
    release_heap();
    data_ = mem;
    cap_ = cap;
  }

  void release_heap() {
    if (on_heap()) std::allocator<T>().deallocate(data_, cap_);
  }

  void destroy_all() {
    clear();
    release_heap();
    data_ = inline_ptr();
    cap_ = N;
  }

  void steal_or_move(SmallVec&& o) noexcept {
    if (o.on_heap()) {
      data_ = o.data_;
      size_ = o.size_;
      cap_ = o.cap_;
      o.data_ = o.inline_ptr();
      o.size_ = 0;
      o.cap_ = N;
    } else {
      data_ = inline_ptr();
      cap_ = N;
      size_ = o.size_;
      for (size_type i = 0; i < size_; ++i) {
        ::new (static_cast<void*>(data_ + i)) T(std::move(o.data_[i]));
        o.data_[i].~T();
      }
      o.size_ = 0;
    }
  }

  T* data_ = inline_ptr();
  size_type size_ = 0;
  size_type cap_ = N;
  alignas(T) unsigned char inline_storage_[N * sizeof(T)];
};

}  // namespace ptcg
