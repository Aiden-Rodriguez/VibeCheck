class EmoteChainAuthenticator:
    def __init__(self, password, step_timeout, hold_seconds, message_seconds):
        self.password = list(password)
        self.step_timeout = step_timeout
        self.hold_seconds = hold_seconds
        self.message_seconds = message_seconds
        self.active = False
        self.succeeded = False
        self.failed = False
        self.step_index = 0
        self.step_started_at = 0.0
        self.held_emote = None
        self.hold_started_at = None
        self.message_until = 0.0

    def start(self, now):
        self.active = True
        self.succeeded = False
        self.failed = False
        self.step_index = 0
        self.step_started_at = now
        self.held_emote = None
        self.hold_started_at = None
        self.message_until = 0.0

    def reset_step(self, now):
        self.step_index = 0
        self.step_started_at = now
        self.held_emote = None
        self.hold_started_at = None

    def update(self, detected_emote, now):
        self._expire_message(now)
        if not self.active or not self.password:
            return

        if now - self.step_started_at > self.step_timeout:
            self.failed = True
            self.message_until = now + self.message_seconds
            self.reset_step(now)
            return

        target = self.current_target
        if detected_emote != target:
            self.held_emote = None
            self.hold_started_at = None
            return

        if self.held_emote != detected_emote:
            self.held_emote = detected_emote
            self.hold_started_at = now
            return

        if now - self.hold_started_at < self.hold_seconds:
            return

        self._advance(now)

    @property
    def current_target(self):
        if not self.active or self.step_index >= len(self.password):
            return None
        return self.password[self.step_index]

    @property
    def progress(self):
        return self.step_index, len(self.password)

    def remaining_seconds(self, now):
        if not self.active:
            return 0.0
        return max(0.0, self.step_timeout - (now - self.step_started_at))

    def held_seconds(self, now):
        if not self.active or self.hold_started_at is None:
            return 0.0
        return min(self.hold_seconds, now - self.hold_started_at)

    def _advance(self, now):
        self.step_index += 1
        self.held_emote = None
        self.hold_started_at = None

        if self.step_index >= len(self.password):
            self.active = False
            self.succeeded = True
            self.failed = False
            self.message_until = now + self.message_seconds
        else:
            self.step_started_at = now

    def _expire_message(self, now):
        if self.message_until and now >= self.message_until:
            self.succeeded = False
            self.failed = False
            self.message_until = 0.0
