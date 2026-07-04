#include "ptcg/card_db.hpp"

#include <vector>

namespace ptcg {

const CardInfo* find_card(int id) {
  // Flat by-id table (ids are small and dense): one bounds check + load on the
  // hottest lookup in the engine. Built once; read-only afterwards, so it is
  // safe to call concurrently once initialized (magic-static init is atomic).
  static const std::vector<const CardInfo*> index = [] {
    int maxId = 0;
    for (const auto& c : card_db())
      if (c.id > maxId) maxId = c.id;
    std::vector<const CardInfo*> v(static_cast<size_t>(maxId) + 1, nullptr);
    for (const auto& c : card_db())
      if (c.id >= 0) v[static_cast<size_t>(c.id)] = &c;
    return v;
  }();
  if (id < 0 || static_cast<size_t>(id) >= index.size()) return nullptr;
  return index[static_cast<size_t>(id)];
}

}  // namespace ptcg
