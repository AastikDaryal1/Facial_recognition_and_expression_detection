"""
api/email_service.py
─────────────────────
Email sending service using SMTP.
Called from auth router when an invite is generated.
"""

import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from config.settings import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, FROM_EMAIL
from utils.logger import get_logger

log = get_logger(__name__)


async def send_invite_email(
    to_email     : str,
    invite_token : str,
    invited_by   : str,
    role         : str,
    org_name     : str = "VisionX",
    frontend_url : str = "",
) -> bool:
    """
    Send an invitation email with a signup link.
    Returns True if sent successfully, False otherwise.
    """

    from config.settings import FRONTEND_URL
    base_url    = frontend_url or FRONTEND_URL
    signup_link = f"{base_url}/signup?token={invite_token}"

    role_labels = {
        "org_admin" : "Organisation Admin",
        "user"      : "Team Member",
    }
    role_label = role_labels.get(role, role)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin:0;padding:0;background:#0f172a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
      <div style="max-width:560px;margin:40px auto;padding:0 20px;">
        <div style="text-align:center;margin-bottom:32px;">
          <h1 style="color:#f1f5f9;font-size:24px;font-weight:700;margin:0;">
            ✦ VisionX
          </h1>
        </div>
        <div style="background:#1e293b;border:1px solid rgba(99,102,241,0.3);border-radius:16px;padding:40px;">
          <h2 style="color:#f1f5f9;font-size:20px;font-weight:700;margin:0 0 8px;">
            You've been invited!
          </h2>
          <p style="color:#64748b;font-size:14px;margin:0 0 24px;">
            <strong style="color:#94a3b8;">{invited_by}</strong> has invited you to join
            <strong style="color:#94a3b8;">{org_name}</strong> as a
            <strong style="color:#a5b4fc;">{role_label}</strong>.
          </p>
          <div style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);border-radius:8px;padding:12px 16px;margin-bottom:28px;text-align:center;">
            <span style="color:#a5b4fc;font-size:13px;font-weight:600;">
              Role: {role_label}
            </span>
          </div>
          <div style="text-align:center;margin-bottom:28px;">
            <a href="{signup_link}"
               style="display:inline-block;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;text-decoration:none;padding:14px 32px;border-radius:8px;font-size:15px;font-weight:600;letter-spacing:0.3px;">
              Create My Account
            </a>
          </div>
          <p style="color:#475569;font-size:12px;text-align:center;margin:0 0 20px;">
            This invite link expires in <strong style="color:#64748b;">48 hours</strong>.
          </p>
          <hr style="border:none;border-top:1px solid rgba(71,85,105,0.3);margin:24px 0;">
          <p style="color:#475569;font-size:12px;margin:0 0 8px;">
            If the button doesn't work, copy and paste this link:
          </p>
          <p style="background:rgba(15,23,42,0.8);border:1px solid rgba(71,85,105,0.3);border-radius:6px;padding:10px;color:#6366f1;font-size:11px;word-break:break-all;margin:0;">
            {signup_link}
          </p>
        </div>
        <p style="color:#334155;font-size:12px;text-align:center;margin-top:24px;">
          If you didn't expect this invite, you can safely ignore this email.
        </p>
      </div>
    </body>
    </html>
    """

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"You've been invited to join {org_name} on VisionX"
        msg["From"]    = f"VisionX <{FROM_EMAIL}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(html_content, "html"))

        await aiosmtplib.send(
            msg,
            hostname  = SMTP_HOST,
            port      = SMTP_PORT,
            username  = SMTP_USER,
            password  = SMTP_PASSWORD,
            start_tls = True,
        )
        log.info("Invite email sent to %s", to_email)
        return True
    except Exception as e:
        log.error("Failed to send invite email to %s: %s", to_email, e)
        return False