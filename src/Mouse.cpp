#include "Mouse.h"

#include "App.h"
#include "MouseEvent.h"
#include "PathWin.h"

namespace pathtracex {
	/** Mouse event stuff **/
	Mouse::Event::Event(Type type, const Mouse& parent) noexcept
		: type(type)
		, leftPressed(parent.leftPressed)
		, rightPressed(parent.rightPressed)
		, x(parent.x)
		, y(parent.y) { }

	Mouse::Event::Type Mouse::Event::getType() const noexcept {
		return type;
	}

	std::pair<int, int> Mouse::Event::getPos() const noexcept {
		return std::make_pair(x, y);
	}

	int Mouse::Event::getPosX() const noexcept {
		return x;
	}

	int Mouse::Event::getPosY() const noexcept {
		return y;
	}

	bool Mouse::Event::leftIsPressed() const noexcept {
		return leftPressed;
	}

	bool Mouse::Event::rightIsPressed() const noexcept {
		return rightPressed;
	}

	/** Mouse stuff **/

	std::pair<int, int> Mouse::getPos() const noexcept {
		return std::make_pair(x, y);
	}

	std::optional<Mouse::RawDelta> Mouse::readRawDelta() noexcept {
		if (rawDeltaBuffer.empty())
			return std::nullopt;

		const Mouse::RawDelta d = rawDeltaBuffer.front();
		rawDeltaBuffer.pop();
		return d;
	}

	int Mouse::getPosX() const noexcept {
		return x;
	}

	int Mouse::getPosY() const noexcept {
		return y;
	}

	bool Mouse::isInWindow() const noexcept {
		return inWindow;
	}

	bool Mouse::leftIsPressed() const noexcept {
		return leftPressed;
	}

	bool Mouse::rightIsPressed() const noexcept {
		return rightPressed;
	}

	std::optional<Mouse::Event> Mouse::read() noexcept {
		if (buffer.size() == 0u)
			return std::nullopt;

		Mouse::Event e = buffer.front();
		buffer.pop();
		return e;
	}

	bool Mouse::isEmpty() const noexcept {
		return buffer.empty();
	}

	void Mouse::flush() noexcept {
		buffer = std::queue<Event>();
	}

	void Mouse::enableRaw() noexcept {
		rawEnabled = true;
	}

	void Mouse::disableRaw() noexcept {
		rawEnabled = false;
	}

	bool Mouse::rawIsEnabled() const noexcept {
		return rawEnabled;
	}

	void Mouse::onMouseMove(int newx, int newy) noexcept {
		auto dx = newx - x;
		auto dy = newy - y;

		x = newx;
		y = newy;

		MouseMovedEvent mme{ float(x), float(y), float(dx), float(dy) };
		App::raiseEvent(mme);

		//buffer.push(Mouse::Event(Mouse::Event::Type::Move, *this));
		//trimBuffer();
	}

	void Mouse::onMouseEnter() noexcept {
		inWindow = true;
		buffer.push(Mouse::Event(Mouse::Event::Type::Enter, *this));
		trimBuffer();
	}

	void Mouse::onMouseLeave() noexcept {
		inWindow = false;
		buffer.push(Mouse::Event(Mouse::Event::Type::Leave, *this));
		trimBuffer();
	}

	void Mouse::onRawDelta(int dx, int dy) noexcept {
		rawDeltaBuffer.push({ dx, dy });
		trimBuffer();
	}

	void Mouse::onLeftPressed(int x, int y) noexcept {
		leftPressed = true;
		
		MouseButtonPressedEvent mbpe{ MouseButtonType::LeftButton };
		App::raiseEvent(mbpe);
	}

	void Mouse::onLeftReleased(int x, int y) noexcept {
		leftPressed = false;
		
		MouseButtonReleasedEvent mbre{ MouseButtonType::LeftButton };
		App::raiseEvent(mbre);
	}

	void Mouse::onRightPressed(int x, int y) noexcept {
		rightPressed = true;
		
		MouseButtonPressedEvent mbpe{ MouseButtonType::RightButton };
		App::raiseEvent(mbpe);
	}

	void Mouse::onRightReleased(int x, int y) noexcept {
		rightPressed = false;
		
		MouseButtonReleasedEvent mbre{ MouseButtonType::RightButton };
		App::raiseEvent(mbre);
	}

	void Mouse::onWheelUp(int x, int y) noexcept {
		buffer.push(Mouse::Event(Mouse::Event::Type::WheelUp, *this));
		trimBuffer();
	}

	void Mouse::onWheelDown(int x, int y) noexcept {
		buffer.push(Mouse::Event(Mouse::Event::Type::WheelDown, *this));
		trimBuffer();
	}

	void Mouse::onWheelDelta(int x, int y, int delta) noexcept {
		wheelDeltaCarry += delta;
		while (wheelDeltaCarry >= WHEEL_DELTA) {
			wheelDeltaCarry -= WHEEL_DELTA;
			onWheelUp(x, y);
		}
		while (wheelDeltaCarry <= -WHEEL_DELTA) {
			wheelDeltaCarry -= WHEEL_DELTA;
			onWheelDown(x, y);
		}
	}

	void Mouse::trimBuffer() noexcept {
		while (buffer.size() > bufferSize) {
			buffer.pop();
		}
	}

	void Mouse::trimRawInputBuffer() noexcept {
		while (rawDeltaBuffer.size() > bufferSize) {
			rawDeltaBuffer.pop();
		}
	}
}