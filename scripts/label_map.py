"""Intent-to-index mapping for AlephBERT classifier.

Single source of truth, shared by training, export, and evaluation scripts.
"""

from __future__ import annotations

INTENT_LABELS: list[str] = [
    "GROCERY_REQUEST",
    "RECIPE_URL",
    "LIST_QUERY",
    "CLEAR_LIST",
    "REMOVE_ITEM",
    "PARTIAL_COMPLETION",
    "GROUP_INFO",
    "GET_INVITE_CODE",
    "CREATE_INVITE",
    "RENAME_GROUP",
    "LEAVE_GROUP",
    "NOTIFICATION_SETTINGS",
    "REVOKE_INVITE",
    "RECIPE_SEARCH",
    "UPDATE_QUANTITY",
    "BUG_REPORT",
    "OTHER",
]

LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(INTENT_LABELS)}
ID2LABEL: dict[int, str] = {i: label for i, label in enumerate(INTENT_LABELS)}
NUM_LABELS: int = len(INTENT_LABELS)

# One-line English descriptions for each intent.
# Used by the HuggingFace model card, the Gradio demo, and the dataset sample.
EN_DESCRIPTIONS: dict[str, str] = {
    "GROCERY_REQUEST": "Add items to the shopping list",
    "RECIPE_URL": "Recipe URL: extract ingredients from a linked recipe",
    "LIST_QUERY": "Show the current shopping list",
    "CLEAR_LIST": "Mark all items as bought; clear the list",
    "REMOVE_ITEM": "Remove a specific item from the list",
    "PARTIAL_COMPLETION": "Mark most items bought except for some",
    "GROUP_INFO": "Show group members and details",
    "GET_INVITE_CODE": "Get the existing group invite code",
    "CREATE_INVITE": "Generate a new group invite code",
    "RENAME_GROUP": "Change the group name",
    "LEAVE_GROUP": "Leave the current group",
    "NOTIFICATION_SETTINGS": "Toggle notification preferences",
    "REVOKE_INVITE": "Cancel or invalidate a group invite code",
    "RECIPE_SEARCH": "Build a shopping list for a known dish",
    "UPDATE_QUANTITY": "Change the quantity of an existing item",
    "BUG_REPORT": "Report a bug or issue with the bot",
    "OTHER": "Conversational or off-topic message; not a shopping intent",
}
