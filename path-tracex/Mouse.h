#pragma once

#include <optional>
#include <queue>

namespace pathtracex {
	class Mouse {
		friend class Window;
	public:
		struct RawDelta {
			int x, y;
		};

		class Event {
		public:
			enum class Type {
				LPress, LRelease,
				RPress, RRelease,
				WheelUp, WheelDown,
				Enter, Leave,
				Move
			};

			Event(Type type, const Mouse& parent) noexcept;

			Type getType() const noexcept;
			std::pair<int, int> getPos() const noexcept;
			int getPosX() const noexcept;
			int getPosY() const noexcept;
			bool leftIsPressed() const noexcept;
			bool rightIsPressed() const noexcept;

		private:
			Type type;
			bool leftPressed, rightPressed;
			int x, y;
		};

		Mouse() = default;
		Mouse(const Mouse&) = delete;
		Mouse& operator=(const Mouse&) = delete;

		std::pair<int, int> getPos() const noexcept;
		std::optional<RawDelta> readRawDelta() noexcept;
		int getPosX() const noexcept;
		int getPosY() const noexcept;
		bool isInWindow() const noexcept;
		bool leftIsPressed() const noexcept;
		bool rightIsPressed() const noexcept;
		std::optional<Mouse::Event> read() noexcept;
		bool isEmpty() const noexcept;

		void flush() noexcept;
		void enableRaw() noexcept;
		void disableRaw() noexcept;
		bool rawIsEnabled() const noexcept;

	private:
		static constexpr size_t bufferSize = 16u;
		int x, y;
		bool leftPressed = false, rightPressed = false;
		bool inWindow = false;
		int wheelDeltaCarry = 0;
		bool rawEnabled = false;
		std::queue<Event> buffer;
		std::queue<RawDelta> rawDeltaBuffer;

		void onMouseMove(int x, int y) noexcept;
		void onMouseLeave() noexcept;
		void onMouseEnter() noexcept;
		void onRawDelta(int dx, int dy) noexcept;
		void onLeftPressed(int x, int y) noexcept;
		void onLeftReleased(int x, int y) noexcept;
		void onRightPressed(int x, int y) noexcept;
		void onRightReleased(int x, int y) noexcept;
		void onWheelUp(int x, int y) noexcept;
		void onWheelDown(int x, int y) noexcept;
		void onWheelDelta(int x, int y, int delta) noexcept;
		void trimBuffer() noexcept;
		void trimRawInputBuffer() noexcept;
	};
}