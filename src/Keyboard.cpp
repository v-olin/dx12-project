#include "Keyboard.h"

#include "App.h"
#include "KeyEvent.h"
#include <Window.h>

namespace pathtracex {

	bool Keyboard::keyIsPressed(unsigned char keycode) const noexcept {
		return keyStates[keycode];
	}

	bool Keyboard::keyIsEmpty() const noexcept {
		return keyBuffer.empty();
	}

	std::optional<char> Keyboard::readChar() noexcept {
		if (charBuffer.size() > 0u)
		{
			unsigned char charcode = charBuffer.front();
			charBuffer.pop();
			return charcode;
		}
		return std::nullopt;
	}

	bool Keyboard::charIsEmpty() const noexcept {
		return charBuffer.empty();
	}

	void Keyboard::flushKey() noexcept {
		keyBuffer = std::queue<Event>();
	}

	void Keyboard::flushChar() noexcept {
		charBuffer = std::queue<char>();
	}

	void Keyboard::flush() noexcept {
		flushKey();
		flushChar();
	}

	void Keyboard::enableAutorepeat() noexcept {
		autorepeatEnabled = true;
	}

	void Keyboard::disableAutorepeat() noexcept {
		autorepeatEnabled = false;
	}

	bool Keyboard::autorepeatIsEnabled() const noexcept {
		return autorepeatEnabled;
	}

	void Keyboard::onKeyPressed(unsigned char keycode) noexcept {
		keyStates[keycode] = true;
		
		KeyPressedEvent kpe{ keycode, false };
		App::raiseEvent(kpe);
	}

	void Keyboard::onKeyReleased(unsigned char keycode) noexcept {
		keyStates[keycode] = false;

		KeyReleasedEvent kre{ keycode };
		App::raiseEvent(kre);
	}

	void Keyboard::onChar(char c) noexcept {
		charBuffer.push(c);
		TrimBuffer(charBuffer);
	}

	void Keyboard::clearState() noexcept {
		keyStates.reset();
	}

	template<typename T>
	void Keyboard::TrimBuffer(std::queue<T>& buffer) noexcept {
		while (buffer.size() > bufferSize) {
			buffer.pop();
		}
	}
}