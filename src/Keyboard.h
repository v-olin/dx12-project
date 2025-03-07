#pragma once

#include "Event.h"

#include <bitset>
#include <optional>
#include <queue>

namespace pathtracex {
	class Keyboard {
		friend class Window; // this is useless???
	public:
		Keyboard() = default;
		Keyboard(const Keyboard&) = delete;
		void operator=(const Keyboard&) = delete;

		bool keyIsPressed(unsigned char keycode) const noexcept;
		//std::optional<Event> readKey() noexcept;
		bool keyIsEmpty() const noexcept;
		void flushKey() noexcept;

		/* Char event stuff */
		std::optional<char> readChar() noexcept;
		bool charIsEmpty() const noexcept;
		void flushChar() noexcept;
		void flush() noexcept;

		/* auto-repeat control */
		void enableAutorepeat() noexcept;
		void disableAutorepeat() noexcept;
		bool autorepeatIsEnabled() const noexcept;

	private:
		void onKeyPressed(unsigned char keycode) noexcept;
		void onKeyReleased(unsigned char keycode) noexcept;
		void onChar(char c) noexcept;
		void clearState() noexcept;
		template<typename T>
		static void TrimBuffer(std::queue<T>& buffer) noexcept;

		static constexpr unsigned int nKeys = 256u;
		static constexpr unsigned int bufferSize = 16u;
		bool autorepeatEnabled = false;
		std::bitset<nKeys> keyStates;
		std::queue<Event> keyBuffer;
		std::queue<char> charBuffer;
	};
}