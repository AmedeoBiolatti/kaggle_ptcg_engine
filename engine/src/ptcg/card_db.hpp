#pragma once
#include "ptcg/card_db.gen.hpp"

namespace ptcg {

// Lookup structural card data by card id; nullptr if not in the DB.
const CardInfo* find_card(int id);

}  // namespace ptcg
