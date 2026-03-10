"""
Two-level category taxonomy for StillOpen.

Maps raw Overture category tags → parent groups.
Based on actual 1.4M+ California Overture dataset category values.
"""

CATEGORY_TREE: dict[str, list[str]] = {
    "Food & Drink": [
        # High-volume actual categories
        "restaurant", "fast_food_restaurant", "coffee_shop", "mexican_restaurant",
        "pizza_restaurant", "sandwich_shop", "cafe", "chinese_restaurant",
        "american_restaurant", "bakery", "bar", "burger_restaurant",
        "sushi_restaurant", "ice_cream_shop", "italian_restaurant",
        "japanese_restaurant", "seafood_restaurant", "taco_restaurant",
        "thai_restaurant", "vietnamese_restaurant", "barbecue_restaurant",
        "chicken_restaurant", "breakfast_and_brunch_restaurant", "bubble_tea",
        "smoothie_juice_bar", "donuts", "winery", "desserts",
        "indian_restaurant", "food_court", "food_truck",
        # Generic fallbacks
        "fast_food", "pizza", "ice_cream", "juice_bar", "brewery",
        "wine_bar", "cocktail_bar", "tea_house", "smoothie_shop",
        "dessert_shop", "candy_store", "chocolate_shop", "donut_shop",
        "bagel_shop", "noodle_house", "ramen_restaurant", "burrito_restaurant",
        "poke_restaurant", "steakhouse", "vegetarian_restaurant",
        "vegan_restaurant", "food_and_drink", "catering", "food_delivery",
        "pub", "diner",
    ],
    "Shopping": [
        "clothing_store", "convenience_store", "grocery_store",
        "building_supply_store", "hardware_store", "liquor_store",
        "furniture_store", "shoe_store", "electronics", "pet_store",
        "sporting_goods", "tobacco_shop", "discount_store",
        "department_store", "gift_shop", "boutique", "antique_store",
        "thrift_store", "supermarket", "shopping_center", "wholesale_store",
        "carpet_store", "nursery_and_gardening", "arts_and_crafts",
        "mobile_phone_store", "retail", "flowers_and_gifts_shop",
        "cosmetic_and_beauty_supplies", "fashion_accessories_store",
        "jewelry_store", "womens_clothing_store", "grocery",
        "pharmacy",       # also Health, but shopping context common
        "bookstore", "toy_store", "bicycle_store", "music_store",
        "art_supply_store", "fabric_store", "florist", "garden_center",
        "optical_store", "home_goods", "kitchen_store",
        "vitamins_supplements_store", "outdoor_clothing",
    ],
    "Health & Wellness": [
        "dentist", "doctor", "gym", "medical_center", "chiropractor",
        "counseling_and_mental_health", "optometrist", "physical_therapy",
        "hospital", "massage_therapy", "massage", "general_dentistry",
        "internal_medicine", "pediatrician", "obstetrician_and_gynecologist",
        "orthopedist", "spas", "acupuncture", "medical_spa",
        "health_and_medical", "yoga_studio", "naturopathic_holistic",
        "skin_care", "martial_arts_club", "home_health_care", "beauty_and_spa",
        "laboratory_testing", "urgent_care", "clinic",
        "dermatologist", "psychiatrist", "psychologist", "counselor",
        "wellness_center", "pilates_studio", "boxing_gym",
        "crossfit_gym", "martial_arts", "meditation_center",
        "blood_bank", "dialysis_center", "rehab_center", "laboratory",
        "mental_health",
    ],
    "Services": [
        "real_estate_agent", "beauty_salon", "hair_salon", "insurance_agency",
        "professional_services", "contractor", "key_and_locksmith",
        "printing_services", "electrician", "plumbing", "hvac_services",
        "car_wash", "dry_cleaning", "laundromat", "landscaping",
        "pest_control_service", "pet_groomer", "home_service", "home_cleaning",
        "courier_and_delivery_services", "shipping_center", "post_office",
        "tax_services", "accountant", "mortgage_broker", "mortgage_lender",
        "financial_service", "financial_advising", "advertising_agency",
        "marketing_agency", "employment_agencies", "event_photography",
        "tattoo_and_piercing", "event_planning", "barber", "nail_salon",
        "real_estate", "property_management", "storage_facility",
        "self_storage_facility", "roofing", "construction_services",
        "garage_door_service", "engineering_services",
        "it_service_and_computer_repair", "software_development",
        "interior_design", "legal_services", "lawyer", "personal_injury_law",
        "money_transfer_services", "atms", "bank_credit_union", "banks",
        "credit_union", "insurance", "real_estate_agent",
        "notary", "photographer", "moving_company", "cleaning_service",
        "locksmith", "kennel", "dog_trainer", "translation_service",
        "financial_advisor", "laundry", "tattoo_parlor",
    ],
    "Automotive": [
        "automotive_repair", "gas_station", "automotive_parts_and_accessories",
        "auto_body_shop", "car_dealer", "tire_dealer_and_repair",
        "auto_insurance", "used_car_dealer", "car_rental_agency", "parking",
        "auto_detailing", "emissions_inspection", "automotive", "truck_rentals",
        "auto_repair", "auto_parts", "tire_shop", "oil_change",
        "towing_service", "motorcycle_dealer", "rv_dealer",
        "auto_glass_repair", "muffler_shop", "transmission_shop",
        "car_inspection", "car_audio", "detailing_service",
    ],
    "Entertainment": [
        "arts_and_entertainment", "sports_and_recreation_venue",
        "movie_theater", "bowling_alley", "arcade", "casino", "nightclub",
        "music_venue", "comedy_club", "escape_room", "mini_golf", "go_kart",
        "trampoline_park", "amusement_park", "water_park", "theme_park",
        "paintball", "laser_tag", "rock_climbing_gym", "axe_throwing",
        "virtual_reality", "karaoke_bar", "sports_bar", "billiards",
        "bingo_hall", "dance_club", "live_music", "concert_hall",
    ],
    "Arts & Culture": [
        "art_gallery", "library", "landmark_and_historical_building",
        "museum", "theater", "cultural_center", "historic_site",
        "science_center", "aquarium", "zoo", "planetarium",
        "botanical_garden", "sculpture_garden", "performing_arts",
        "opera_house", "ballet", "art_studio", "pottery_studio",
        "photography_gallery", "antique_store", "auction_house",
    ],
    "Education": [
        "elementary_school", "preschool", "high_school", "school",
        "college_university", "tutoring_center", "education", "dance_school",
        "university", "college", "tutoring", "daycare",
        "driving_school", "language_school", "vocational_school",
        "trade_school", "coding_school", "music_school", "art_school",
        "cooking_school", "sports_academy", "test_prep", "special_education",
    ],
    "Travel & Lodging": [
        "hotel", "motel", "resort", "campground", "car_rental_agency",
        "travel", "travel_services", "hostel", "vacation_rental",
        "airport", "bed_and_breakfast", "inn", "extended_stay", "rv_park",
        "bus_station", "train_station", "taxi", "rideshare", "tour_operator",
        "marina",
    ],
    "Outdoors & Recreation": [
        "park", "lake", "farm", "sports_complex", "golf_course",
        "tennis_court", "swimming_pool", "ski_resort", "nature_reserve",
        "wildlife_sanctuary", "fishing_spot", "bike_path", "skate_park",
        "basketball_court", "soccer_field", "baseball_field",
        "football_field", "running_track", "disc_golf", "horseback_riding",
        "kayaking", "surfing", "diving", "rock_climbing", "playground",
        "beach", "hiking_trail",
    ],
    "Community": [
        "church_cathedral", "community_services_non_profits",
        "public_and_government_association", "fire_department",
        "social_service_organizations", "religious_organization",
        "baptist_church", "retirement_home",
        "church", "mosque", "temple", "synagogue", "community_center",
        "nonprofit", "fire_station", "police_station", "city_hall",
        "food_bank", "homeless_shelter", "senior_center", "youth_center",
        "veterans_organization", "civic_organization", "union_hall",
        "apartments",
    ],
    "Other": [],
}

# Pre-compute the set of all explicitly mapped raw categories for fast lookup
_ALL_MAPPED: frozenset[str] = frozenset(
    cat
    for parent, children in CATEGORY_TREE.items()
    if parent != "Other"
    for cat in children
)

# Build reverse map: raw_category → parent
_REVERSE: dict[str, str] = {}
for _parent, _children in CATEGORY_TREE.items():
    if _parent == "Other":
        continue
    for _child in _children:
        _REVERSE[_child.lower()] = _parent


def get_parent_category(raw_category: str) -> str:
    """Return the parent group for a raw Overture/OSM category tag."""
    if not raw_category:
        return "Other"
    return _REVERSE.get(raw_category.lower(), "Other")


def get_children(parent: str) -> list[str]:
    """Return raw category tags that belong to a parent group."""
    return CATEGORY_TREE.get(parent, [])


def all_mapped_categories() -> frozenset[str]:
    """Return the set of all raw categories that map to a non-Other parent."""
    return _ALL_MAPPED
