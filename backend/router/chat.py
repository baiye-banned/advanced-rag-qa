from fastapi import APIRouter

from service.chat_export_service import export_conversation
from service.trace_service import get_chat_trace, get_message_trace
from service.chat_service import (
    delete_conversation,
    delete_message,
    get_messages,
    list_conversations,
    rename_conversation,
    submit_message_feedback,
    upload_chat_attachment,
    stream_chat,
)


router = APIRouter()
router.add_api_route("/api/chat/conversations", list_conversations, methods=["GET"])
router.add_api_route("/api/chat/conversations/{cid}", get_messages, methods=["GET"])
router.add_api_route("/api/chat/conversations/{cid}/export", export_conversation, methods=["GET"])
router.add_api_route("/api/chat/traces/{trace_id}", get_chat_trace, methods=["GET"])
router.add_api_route("/api/chat/messages/{message_id}/trace", get_message_trace, methods=["GET"])
router.add_api_route("/api/chat/messages/{message_id}", delete_message, methods=["DELETE"])
router.add_api_route("/api/chat/conversations/{cid}", delete_conversation, methods=["DELETE"])
router.add_api_route("/api/chat/conversations/{cid}", rename_conversation, methods=["PUT"])
router.add_api_route("/api/chat/attachments", upload_chat_attachment, methods=["POST"])
router.add_api_route("/api/chat/stream", stream_chat, methods=["POST"])
router.add_api_route(
    "/api/chat/messages/{message_id}/feedback", submit_message_feedback, methods=["POST"]
)
